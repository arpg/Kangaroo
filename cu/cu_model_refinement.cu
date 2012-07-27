#include "all.h"
#include "launch_utils.h"

namespace Gpu {

__host__ __device__ inline
float LSReweightSq(float r, float c) {
    return 1;
}

__host__ __device__ inline
float LSReweightL1(float r, float c) {
    const float absr = abs(r);
    return 1.0f / absr;
}

__host__ __device__ inline
float LSReweightHuber(float r, float c) {
    const float absr = abs(r);
    return (absr <= c ) ? 1.0f : c / absr;
}

__host__ __device__ inline
float LSReweightTukey(float r, float c) {
    const float absr = abs(r);
    const float roc = r / c;
    const float omroc2 = 1.0f - roc*roc;
    return (absr <= c ) ? omroc2*omroc2 : 0.0f;
}

template<typename Ti>
__global__ void KernPoseRefinementFromDepthmap(
    const Image<Ti> dImgl,
    const Image<Ti> dImgr, const Image<float4> dPr,
    const Mat<float,3,4> KT_lr, float c,
    Image<LeastSquaresSystem<float,6> > dSum, Image<float4> dDebug
){
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;
//    const Mat<float,2> pr = {u,v};

    LeastSquaresSystem<float,6> sum;
    sum.SetZero();

    const float4 Pr4 = dPr(u,v);
    // OpenGL point to what our intrinsics expects
    const Mat<float,4> Pr = {Pr4.x, -Pr4.y, -Pr4.z, 1};

    const Mat<float,3> KPl = KT_lr * Pr;
    const Mat<float,2> pl = {KPl(0)/KPl(2), KPl(1)/KPl(2)};

    if(isfinite(Pr4.z) && dImgl.InBounds(pl(0), pl(1), 2)) {

        float Il = dImgl.template GetBilinear<float>(pl(0), pl(1));
        float Ir = dImgr(u,v);
        const float y = Il - Ir;

        const Mat<float,1,2> dIl = dImgl.template GetCentralDiff<float>(pl(0), pl(1));

        const Mat<float,2,3> dPl_by_dpl = {
          1.0/KPl(2), 0, -KPl(0)/(KPl(2)*KPl(2)),
          0, 1.0/KPl(2), -KPl(1)/(KPl(2)*KPl(2))
        };

        const Mat<float,1,4> dIldPlKT_lr = dIl * dPl_by_dpl * KT_lr;

        // Sparse Jr_i = dIl * dPl_by_dpl * KT_lr * gen_i * Pr
        const Mat<float,1,6> Jr = {
            dIldPlKT_lr(0),
            dIldPlKT_lr(1),
            dIldPlKT_lr(2),
            -dIldPlKT_lr(1)*Pr(2) + dIldPlKT_lr(2)*Pr(1),
            +dIldPlKT_lr(0)*Pr(2) - dIldPlKT_lr(2)*Pr(0),
            -dIldPlKT_lr(0)*Pr(1) + dIldPlKT_lr(1)*Pr(0)
        };

        const float w = LSReweightTukey(y,c);
        sum.JTJ = OuterProduct(Jr,w);
        sum.JTy = Jr * (y*w);
        sum.obs = 1;
        sum.sqErr = y*y;

        const float debug = (abs(y) + 128) / 255.0f;
        dDebug(u,v) = make_float4(debug,0,w,1);
    }else{
        dDebug(u,v) = make_float4(1,0,0,1);
    }

    dSum(u,v) = sum;
}

LeastSquaresSystem<float,6> PoseRefinementFromDepthmap(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float4> dPr,
    const Mat<float,3,4> KT_lr, float c,
    Image<unsigned char> dWorkspace, Image<float4> dDebug
){
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dImgr);
    Image<LeastSquaresSystem<float,6> > dSum = dWorkspace.PackedImage<LeastSquaresSystem<float,6> >(dImgr.w, dImgr.h);

    KernPoseRefinementFromDepthmap<unsigned char><<<gridDim,blockDim>>>(dImgl, dImgr, dPr, KT_lr, c, dSum, dDebug );

    LeastSquaresSystem<float,6> sum;
    sum.SetZero();
    return thrust::reduce(dSum.begin(), dSum.end(), sum, thrust::plus<LeastSquaresSystem<float,6> >() );
}

__global__ void KernPoseRefinementProjectiveIcpPointPlane(
    const Image<float4> dPl,
    const Image<float4> dPr, const Image<float4> dNr,
    const Mat<float,3,4> KT_lr, const Mat<float,3,4> T_rl, float c,
    Image<LeastSquaresSystem<float,6> > dSum, Image<float4> dDebug
) {
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    LeastSquaresSystem<float,6> sum;
    sum.SetZero();

    dSum(u,v) = sum;
    dDebug(u,v) = make_float4(u/(float)dPl.w,v/(float)dPl.h,0,1);
}


LeastSquaresSystem<float,6> PoseRefinementProjectiveIcpPointPlane(
    const Image<float4> dPl,
    const Image<float4> dPr, const Image<float4> dNr,
    const Mat<float,3,4> KT_lr, const Mat<float,3,4> T_rl, float c,
    Image<unsigned char> dWorkspace, Image<float4> dDebug
){
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dPl);
    Image<LeastSquaresSystem<float,6> > dSum = dWorkspace.PackedImage<LeastSquaresSystem<float,6> >(dPl.w, dPl.h);

    KernPoseRefinementProjectiveIcpPointPlane<<<gridDim,blockDim>>>(dPl, dPr, dNr, KT_lr, T_rl, c, dSum, dDebug );

    LeastSquaresSystem<float,6> sum;
    sum.SetZero();
    return thrust::reduce(dSum.begin(), dSum.end(), sum, thrust::plus<LeastSquaresSystem<float,6> >() );
}


}

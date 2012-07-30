#include "all.h"
#include "launch_utils.h"
#include "reweighting.h"

namespace Gpu {

//////////////////////////////////////////////////////
// Pose refinement from depthmap
//////////////////////////////////////////////////////

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
    const Mat<float,4> Pr = {Pr4.x, Pr4.y, Pr4.z, 1};

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

        // Sparse Jr_i = dIldPlKT_lr * gen_i * Pr
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
        sum.JTy = mul_aTb(Jr, y*w);
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

//////////////////////////////////////////////////////
// Projective ICP with Point Plane constraint
//////////////////////////////////////////////////////

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

    const float4 Pr = dPr(u,v);
    const float4 Nr = dNr(u,v);

    const float3 KPl = KT_lr * Pr;
    const float2 pl = dn(KPl);

    if( isfinite(Pr.z) && isfinite(Nr.z) && dPl.InBounds(pl, 2) ) {
        const float4 _Pl = dPl.GetNearestNeighbour(pl);
        if(isfinite(_Pl.z)) {
            const float3 _Pr = T_rl * _Pl;
            const float3 Dr = _Pr - Pr;
            const float DrDotNr = dot(Dr,Nr);
            const float y = DrDotNr;

            const Mat<float,1,6> Jr = {
                dot(-1.0*SE3gen0mul(_Pr), Nr),
                dot(-1.0*SE3gen1mul(_Pr), Nr),
                dot(-1.0*SE3gen2mul(_Pr), Nr),
                dot(-1.0*SE3gen3mul(_Pr), Nr),
                dot(-1.0*SE3gen4mul(_Pr), Nr),
                dot(-1.0*SE3gen5mul(_Pr), Nr)
            };

            const float w = LSReweightTukey(y,c);
            sum.JTJ = OuterProduct(Jr,w);
            sum.JTy = mul_aTb(Jr,y*w);
            sum.obs = 1;
            sum.sqErr = y*y;

            const float db = abs(y);
            dDebug(u,v) = make_float4(db,db,db,1);
        }else{
            dDebug(u,v) = make_float4(0,0,1,1);
        }
    }else{
        dDebug(u,v) = make_float4(1,0,0,1);
    }

    dSum(u,v) = sum;
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

//////////////////////////////////////////////////////
// Kinect Calibration
//////////////////////////////////////////////////////

template<typename TI>
__global__ void KernKinectCalibration(
    const Image<float4> dPl, const Image<TI> dIl,
    const Image<float4> dPr, const Image<TI> dIr,
    const Mat<float,3,4> KcT_cd, const Mat<float,3,4> T_lr,
    float c, Image<LeastSquaresSystem<float,2*6> > dSum, Image<float4> dDebug
){
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    LeastSquaresSystem<float,2*6> sum;
    sum.SetZero();

    const float4 Pr = dPr(u,v);
    const float3 Pl = T_lr * Pr;
    const float3 _pl = KcT_cd * Pl;
    const float3 _pr = KcT_cd * Pr;
    const float2 pl = dn(_pl);
    const float2 pr = dn(_pr);

    if( dIl.InBounds(pl,2) && dIr.InBounds(pr,2) ) {
        const float3 y = dIl.template GetBilinear<float3>(pl) - dIr.template GetBilinear<float3>(pr);

        const Mat<float3,1,2> dIldxy = dIl.template GetCentralDiff<float3>(pl.x, pl.y);
        const Mat<float,2,3> dpld_pl = {
          1.0/_pl.z, 0, -_pl.x/(_pl.z*_pl.z),
          0, 1.0/_pl.z, -_pl.y/(_pl.z*_pl.z)
        };
        const Mat<float,2,4> dpld_plKcT_cd = dpld_pl * KcT_cd;
        const Mat<float3,1,4> dIldxydpld_plKcT_cd = dIldxy * dpld_plKcT_cd;
        const Mat<float3,1,4> dIldxydpld_plKcT_cdT_lr = dIldxy * (dpld_pl * (KcT_cd * T_lr) );

        const Mat<float3,1,2> dIrdxy = dIr.template GetCentralDiff<float3>(pr.x, pr.y);
        const Mat<float,2,3> dprd_pr = {
            1.0/_pr.z, 0, -_pr.x/(_pr.z*_pr.z),
            0, 1.0/_pr.z, -_pr.y/(_pr.z*_pr.z)
        };

        const Mat<float3,1,4> dIrdxydprd_prKcT_cd = dIrdxy * (dprd_pr * KcT_cd);

        const Mat<float3,1,12> Jr = {
            dIldxydpld_plKcT_cd * SE3gen0mul(Pl) - dIrdxydprd_prKcT_cd * SE3gen0mul(Pr),
            dIldxydpld_plKcT_cd * SE3gen1mul(Pl) - dIrdxydprd_prKcT_cd * SE3gen1mul(Pr),
            dIldxydpld_plKcT_cd * SE3gen2mul(Pl) - dIrdxydprd_prKcT_cd * SE3gen2mul(Pr),
            dIldxydpld_plKcT_cd * SE3gen3mul(Pl) - dIrdxydprd_prKcT_cd * SE3gen3mul(Pr),
            dIldxydpld_plKcT_cd * SE3gen4mul(Pl) - dIrdxydprd_prKcT_cd * SE3gen4mul(Pr),
            dIldxydpld_plKcT_cd * SE3gen5mul(Pl) - dIrdxydprd_prKcT_cd * SE3gen5mul(Pr),
            dIldxydpld_plKcT_cdT_lr * SE3gen0mul(Pr),
            dIldxydpld_plKcT_cdT_lr * SE3gen1mul(Pr),
            dIldxydpld_plKcT_cdT_lr * SE3gen2mul(Pr),
            dIldxydpld_plKcT_cdT_lr * SE3gen3mul(Pr),
            dIldxydpld_plKcT_cdT_lr * SE3gen4mul(Pr),
            dIldxydpld_plKcT_cdT_lr * SE3gen5mul(Pr)
        };

        const float w = LSReweightTukey(y.x,c)+LSReweightTukey(y.y,c)+LSReweightTukey(y.z,c);
        sum.JTJ = OuterProduct(Jr,w);
        sum.JTy = mul_aTb(Jr, y*w);
        sum.obs = 1;
        sum.sqErr = dot(y,y);

        const float f = abs(y.x) + abs(y.y) + abs(y.z);
        const float d = f/(3*255.0f);
        dDebug(u,v) = make_float4(d,d,d,1);
    }else{
        dDebug(u,v) = make_float4(1,0,0,1);
    }
    dSum(u,v) = sum;
}

LeastSquaresSystem<float,2*6> KinectCalibration(
    const Image<float4> dPl, const Image<uchar3> dIl,
    const Image<float4> dPr, const Image<uchar3> dIr,
    const Mat<float,3,4> KcT_cd, const Mat<float,3,4> T_lr,
    float c, Image<unsigned char> dWorkspace, Image<float4> dDebug
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dPl);
    Image<LeastSquaresSystem<float,2*6> > dSum = dWorkspace.PackedImage<LeastSquaresSystem<float,2*6> >(dPl.w, dPl.h);

    KernKinectCalibration<uchar3><<<gridDim,blockDim>>>(dPl, dIl, dPr, dIr, KcT_cd, T_lr, c, dSum, dDebug );

    LeastSquaresSystem<float,2*6> sum;
    sum.SetZero();
    return thrust::reduce(dSum.begin(), dSum.end(), sum, thrust::plus<LeastSquaresSystem<float,2*6> >() );

}

}

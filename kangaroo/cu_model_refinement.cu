#include "MatUtils.h"
#include "Image.h"
#include "launch_utils.h"
#include "reweighting.h"
#include "disparity.h"
#include "LeastSquareSum.h"

namespace Gpu {

///////////////////////////////////////////
// IsFinite
///////////////////////////////////////////

template<unsigned R, unsigned C>
__device__ inline
bool IsFinite(const Mat<float,R,C>& mat)
{
    for(int i=0; i<R*C; ++i) {
        if( !isfinite(mat[i]) )
            return false;
    }
    return true;
}

template<unsigned R, unsigned C>
__device__ inline
bool IsFinite(const Mat<float3,R,C>& mat)
{
    for(int i=0; i<R*C; ++i) {
        const float3& el = mat[i];
        if( !isfinite(el.x) || !isfinite(el.y) || !isfinite(el.z) )
            return false;
    }
    return true;
}

//////////////////////////////////////////////////////
// Pose refinement from depthmap
//////////////////////////////////////////////////////

template<typename Ti>
__device__ inline
void BuildPoseRefinementFromDepthmapSystem(
    const unsigned int u,  const unsigned int v, const float4 Pr4,
    const Image<Ti>& dImgl, const Image<Ti>& dImgr,
    const Mat<float,3,4>& KT_lr, float c,
    LeastSquaresSystem<float,6>& lss, Image<float4> dDebug
) {
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
        lss.JTJ = OuterProduct(Jr,w);
        lss.JTy = mul_aTb(Jr, y*w);
        lss.obs = 1;
        lss.sqErr = y*y;

        const float debug = (abs(y) + 128) / 255.0f;
        dDebug(u,v) = make_float4(debug,0,w,1);
//        dDebug(u,v) = make_float4(debug,debug,debug,1);
//        dDebug(u,v) = make_float4(0.5 + dIl(0)/100.0,0.5 + dIl(1)/100.0, 0,1);
//        dDebug(u,v) = make_float4(1.0/Pr4.z,1.0/Pr4.z,1.0/Pr4.z,1);
    }else{
        dDebug(u,v) = make_float4(1,0,0,1);
    }
}

template<typename Ti>
__device__ inline
void BuildPoseRefinementFromDepthmapSystemESM(
    const unsigned int u,  const unsigned int v, const float4 Pr4,
    const Image<Ti>& dImgl, const Image<Ti>& dImgr,
    const float fu, const float fv, const float u0, const float v0,
    const Mat<float,3,4>& KT_lr, float c,
    LeastSquaresSystem<float,6>& lss, Image<float4> dDebug, const bool bDiscardMaxMin
) {
    const Mat<float,4> Pr = {Pr4.x, Pr4.y, Pr4.z, 1};
    Mat<float,3> KPr;

    KPr(0) = fu*Pr(0) + u0*Pr(2);
    KPr(1) = fv*Pr(1) + v0*Pr(2);
    KPr(2) = Pr(2);

    const Mat<float,3> KPl = KT_lr * Pr;
    const Mat<float,2> pl = {KPl(0)/KPl(2), KPl(1)/KPl(2)};

    if(isfinite(Pr4.z) && dImgl.InBounds(pl(0), pl(1), 2)) {

        float Il = dImgl.template GetBilinear<float>(pl(0), pl(1));
        float Ir = dImgr(u,v);

        if( bDiscardMaxMin && ( Il == 0 || Il == 255 || Ir == 0 || Ir == 255 ) ) {
            dDebug(u,v) = make_float4(1,1,0,1);
        } else {
            const float y = Il - Ir;

            const Mat<float,1,2> dIl = dImgl.template GetCentralDiff<float>(pl(0), pl(1));
            const Mat<float,1,2> dIr = dImgr.template GetCentralDiff<float>((float)u, (float)v);

            const Mat<float,2,3> dPl_by_dpl = {
              1.0/KPl(2), 0, -KPl(0)/(KPl(2)*KPl(2)),
              0, 1.0/KPl(2), -KPl(1)/(KPl(2)*KPl(2))
            };

            const Mat<float,2,3> dPr = {
              1.0/KPr(2), 0, -KPr(0)/(KPr(2)*KPr(2)),
              0, 1.0/KPr(2), -KPr(1)/(KPr(2)*KPr(2))
            };

            const Mat<float,1,4> dIldPlKT_lr = dIl * dPl_by_dpl * KT_lr;
            const Mat<float,1,3> dIrdPr = dIr * dPr;

            // Sparse Jl = dIldPlKT_lr * gen_i * Pr
            const Mat<float,1,6> Jl = {
                dIldPlKT_lr(0),
                dIldPlKT_lr(1),
                dIldPlKT_lr(2),
                -dIldPlKT_lr(1)*Pr(2) + dIldPlKT_lr(2)*Pr(1),
                +dIldPlKT_lr(0)*Pr(2) - dIldPlKT_lr(2)*Pr(0),
                -dIldPlKT_lr(0)*Pr(1) + dIldPlKT_lr(1)*Pr(0)
            };

            // Sparse Jr = dIrdPrK * gen_i * KPr
            const Mat<float,1,6> Jr = {
                dIrdPr(0),
                dIrdPr(1),
                dIrdPr(2),
                -dIrdPr(1)*KPr(2) + dIrdPr(2)*KPr(1),
                +dIrdPr(0)*KPr(2) - dIrdPr(2)*KPr(0),
                -dIrdPr(0)*KPr(1) + dIrdPr(1)*KPr(0)
            };


            // ESM Jacobian
            const Mat<float,1,6> J = {
                (Jr(0) + Jl(0))/2,
                (Jr(1) + Jl(1))/2,
                (Jr(2) + Jl(2))/2,
                (Jr(3) + Jl(3))/2,
                (Jr(4) + Jl(4))/2,
                (Jr(5) + Jl(5))/2
            };

            const float w = LSReweightTukey(y,c);
            lss.JTJ = OuterProduct(Jl,w);
            lss.JTy = mul_aTb(Jl, y*w);
            lss.obs = 1;
            lss.sqErr = y*y;

            const float debug = (abs(y) + 128) / 255.0f;
            dDebug(u,v) = make_float4(debug,0,w,1);
    //        dDebug(u,v) = make_float4(debug,debug,debug,1);
    //        dDebug(u,v) = make_float4(0.5 + dIl(0)/100.0,0.5 + dIl(1)/100.0, 0,1);
    //        dDebug(u,v) = make_float4(1.0/Pr4.z,1.0/Pr4.z,1.0/Pr4.z,1);
        }
    }else{
        dDebug(u,v) = make_float4(0,1,0,1);
    }
}


template<typename Ti>
__global__ void KernPoseRefinementFromVbo(
    const Image<Ti> dImgl, const Image<Ti> dImgr, const Image<float4> dPr,
    const Mat<float,3,4> KT_lr, float c,
    Image<LeastSquaresSystem<float,6> > dSum, Image<float4> dDebug
){
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    __shared__ SumLeastSquaresSystem<float,6,16,16> sumlss;

    const float4 Pr4 = dPr(u,v);
    BuildPoseRefinementFromDepthmapSystem(u,v,Pr4,dImgl,dImgr,KT_lr,c,sumlss.ZeroThisObs(),dDebug);

    sumlss.ReducePutBlock(dSum);
}

LeastSquaresSystem<float,6> PoseRefinementFromVbo(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float4> dPr,
    const Mat<float,3,4> KT_lr, float c,
    Image<unsigned char> dWorkspace, Image<float4> dDebug
){
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dImgr, 16, 16);

    HostSumLeastSquaresSystem<float,6> lss(dWorkspace, blockDim, gridDim);
    KernPoseRefinementFromVbo<unsigned char><<<gridDim,blockDim>>>(dImgl, dImgr, dPr, KT_lr, c, lss.LeastSquareImage(), dDebug );
    return lss.FinalSystem();
}

template<typename Ti>
__global__ void KernPoseRefinementFromDisparity(
    const Image<Ti> dImgl, const Image<Ti> dImgr, const Image<float> dDispr,
    const Mat<float,3,4> KT_lr, float c,
    float baseline, float fu, float fv, float u0, float v0,
    Image<LeastSquaresSystem<float,6> > dSum, Image<float4> dDebug
){
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    __shared__ SumLeastSquaresSystem<float,6,16,16> lss;

    const float4 Pr4 = DepthFromDisparity(u,v,dDispr(u,v), baseline, fu, fv, u0, v0);
    BuildPoseRefinementFromDepthmapSystem(u,v,Pr4,dImgl,dImgr,KT_lr,c,lss.ZeroThisObs(),dDebug);

    lss.ReducePutBlock(dSum);
}

LeastSquaresSystem<float,6> PoseRefinementFromDisparity(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float> dDispr,
    const Mat<float,3,4> KT_lr, float c,
    float baseline, float fu, float fv, float u0, float v0,
    Image<unsigned char> dWorkspace, Image<float4> dDebug
){
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dImgr, 16, 16);

    HostSumLeastSquaresSystem<float,6> lss(dWorkspace, blockDim, gridDim);
    KernPoseRefinementFromDisparity<unsigned char><<<gridDim,blockDim>>>(dImgl, dImgr, dDispr, KT_lr, c, baseline, fu, fv, u0, v0, lss.LeastSquareImage(), dDebug );
    return lss.FinalSystem();
}

template<typename Ti>
__global__ void KernPoseRefinementFromDisparityESM(
    const Image<Ti> dImgl, const Image<Ti> dImgr, const Image<float> dDispr,
    const Mat<float,3,4> KT_lr, float c,
    float baseline, float fu, float fv, float u0, float v0,
    Image<LeastSquaresSystem<float,6> > dSum, Image<float4> dDebug,
    const bool bDiscardMaxMin
){
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    __shared__ SumLeastSquaresSystem<float,6,16,16> lss;

    const float4 Pr4 = DepthFromDisparity(u,v,dDispr(u,v), baseline, fu, fv, u0, v0 );
    BuildPoseRefinementFromDepthmapSystemESM(u,v,Pr4,dImgl,dImgr,fu, fv, u0, v0, KT_lr,c,lss.ZeroThisObs (),dDebug,bDiscardMaxMin);

    lss.ReducePutBlock(dSum);
}

LeastSquaresSystem<float,6> PoseRefinementFromDisparityESM(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float> dDispr,
    const Mat<float,3,4> KT_lr, float c,
    float baseline, float fu, float fv, float u0, float v0,
    Image<unsigned char> dWorkspace, Image<float4> dDebug,
    const bool bDiscardMaxMin
){
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dImgr, 16, 16);

    HostSumLeastSquaresSystem<float,6> lss(dWorkspace, blockDim, gridDim);
    KernPoseRefinementFromDisparityESM<unsigned char><<<gridDim,blockDim>>>(dImgl, dImgr, dDispr, KT_lr, c, baseline, fu, fv, u0, v0, lss.LeastSquareImage(), dDebug, bDiscardMaxMin );
    return lss.FinalSystem();
}


template<typename Ti>
__global__ void KernPoseRefinementFromDepthESM(
    const Image<Ti> dImgl, const Image<Ti> dImgr, const Image<float> dDepth,
    const Mat<float,3,4> KT_lr, float c,
    float fu, float fv, float u0, float v0,
    Image<LeastSquaresSystem<float,6> > dSum, Image<float4> dDebug,
    const bool bDiscardMaxMin
){
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    __shared__ SumLeastSquaresSystem<float,6,16,16> lss;

    float4 Pr4;
    Pr4.z = dDepth(u,v);
    Pr4.x = Pr4.z * (u-u0) / fu;
    Pr4.y = Pr4.z * (v-v0) / fv;
    Pr4.w = 1;

    BuildPoseRefinementFromDepthmapSystemESM(u,v,Pr4,dImgl,dImgr,fu, fv, u0, v0, KT_lr,c,lss.ZeroThisObs (),dDebug,bDiscardMaxMin);

    lss.ReducePutBlock(dSum);
}


LeastSquaresSystem<float,6> PoseRefinementFromDepthESM(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float> dDispr,
    const Mat<float,3,4> KT_lr, float c,
    float fu, float fv, float u0, float v0,
    Image<unsigned char> dWorkspace, Image<float4> dDebug,
    const bool bDiscardMaxMin
){
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dImgr, 16, 16);

    HostSumLeastSquaresSystem<float,6> lss(dWorkspace, blockDim, gridDim);
    KernPoseRefinementFromDepthESM<unsigned char><<<gridDim,blockDim>>>(dImgl, dImgr, dDispr, KT_lr, c, fu, fv, u0, v0, lss.LeastSquareImage(), dDebug, bDiscardMaxMin );
    return lss.FinalSystem();
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

    __shared__ SumLeastSquaresSystem<float,6,16,16> sumlss;
    LeastSquaresSystem<float,6>& sum = sumlss.ZeroThisObs();

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

    sumlss.ReducePutBlock(dSum);
}


LeastSquaresSystem<float,6> PoseRefinementProjectiveIcpPointPlane(
    const Image<float4> dPl,
    const Image<float4> dPr, const Image<float4> dNr,
    const Mat<float,3,4> KT_lr, const Mat<float,3,4> T_rl, float c,
    Image<unsigned char> dWorkspace, Image<float4> dDebug
){
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dPl, 16, 16);

    HostSumLeastSquaresSystem<float,6> lss(dWorkspace, blockDim, gridDim);
    KernPoseRefinementProjectiveIcpPointPlane<<<gridDim,blockDim>>>(dPl, dPr, dNr, KT_lr, T_rl, c, lss.LeastSquareImage(), dDebug );
    return lss.FinalSystem();
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

    if( isfinite(Pr.z) && isfinite(Pl.z) && dIl.InBounds(pl,2) && dIr.InBounds(pr,2) ) {
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


//////////////////////////////////////////////////////
// Speed test
//////////////////////////////////////////////////////

const int TEST_SYS_SIZE = 6;

__global__ void KernSumSpeedTest(
    Image<LeastSquaresSystem<float,TEST_SYS_SIZE> > dSum
) {
    __shared__ SumLeastSquaresSystem<float,TEST_SYS_SIZE,16,16> sumlss;
    LeastSquaresSystem<float,TEST_SYS_SIZE>& sum = sumlss.ThisObs();

    sum.SetZero();
    sum.obs = 1;

    sumlss.ReducePutBlock(dSum);
}

void SumSpeedTest(
    Image<unsigned char> dWorkspace, int w, int h, int blockx, int blocky
) {
    dim3 blockDim, gridDim;
    blockDim = dim3(boost::math::gcd<unsigned>(w,blockx), boost::math::gcd<unsigned>(h,blocky), 1);
    gridDim =  dim3( w / blockDim.x, h / blockDim.y, 1);

    HostSumLeastSquaresSystem<float,TEST_SYS_SIZE> lss(dWorkspace, blockDim, gridDim);
    KernSumSpeedTest<<<gridDim,blockDim>>>(lss.LeastSquareImage());
    LeastSquaresSystem<float,TEST_SYS_SIZE> sum = lss.FinalSystem();
    std::cout << sum.obs << std::endl;
}

}

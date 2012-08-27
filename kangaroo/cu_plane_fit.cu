#include "kangaroo.h"
#include "launch_utils.h"
#include "LeastSquareSum.h"

namespace Gpu
{

__global__ void KernPlaneFitGN(const Image<float4> dVbo, const Mat<float,3,3> Qinv, const Mat<float,3> zhat, Image<LeastSquaresSystem<float,3> > dSum, Image<float> dErr, float within, float c )
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    const float4 P = dVbo(x,y);
    __shared__ SumLeastSquaresSystem<float,3,32,32> lss;
    LeastSquaresSystem<float,3>& sum = lss.ThisObs();

    if( isfinite(P.z) && length(P) < within ) {
        const Mat<float,3,1> nhat = Qinv * zhat;
        const float dinv = sqrt(nhat * nhat);
        const float d = 1.0 / dinv;

        const float np_p1 = nhat[0] * P.x + nhat[1] * P.y + nhat[2] * P.z + 1;
        const float y = d * np_p1;
        const float absy = abs(y);
        const float roc = y/c;
        const float omrocsq = (1-roc*roc);
        const float w = (absy <= c) ? omrocsq*omrocsq : 0;

        const Mat<float,3,1> dn_dz0 = zhat[0] * (Mat<float,3,1>){Qinv(0,0), Qinv(1,0), Qinv(2,0)};
        const Mat<float,3,1> dn_dz1 = zhat[1] * (Mat<float,3,1>){Qinv(0,1), Qinv(1,1), Qinv(2,1)};
        const Mat<float,3,1> dn_dz2 = zhat[2] * (Mat<float,3,1>){Qinv(0,2), Qinv(1,2), Qinv(2,2)};

        Mat<float,1,3> Ji;
        Ji[0] = ((-d*d*d*np_p1) * (nhat * dn_dz0)) + d * (dn_dz0[0]*P.x + dn_dz0[1]*P.y + dn_dz0[2]*P.z);
        Ji[1] = ((-d*d*d*np_p1) * (nhat * dn_dz1)) + d * (dn_dz1[0]*P.x + dn_dz1[1]*P.y + dn_dz1[2]*P.z);
        Ji[2] = ((-d*d*d*np_p1) * (nhat * dn_dz2)) + d * (dn_dz2[0]*P.x + dn_dz2[1]*P.y + dn_dz2[2]*P.z);

        sum.JTJ = OuterProduct(Ji, w);
        sum.JTy = mul_aTb(Ji, y*w );
        sum.sqErr = y*y;
        sum.obs = 1;
    }else{
        sum.SetZero();
    }

    dErr(x,y) = sum.sqErr;
    lss.ReducePutBlock(dSum);
}

LeastSquaresSystem<float,3> PlaneFitGN(const Image<float4> dVbo, const Mat<float,3,3> Qinv, const Mat<float,3> zhat, Image<unsigned char> dWorkspace, Image<float> dErr, float within, float c )
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dVbo, 32, 32);

    HostSumLeastSquaresSystem<float,3> lss(dWorkspace, blockDim, gridDim);
    KernPlaneFitGN<<<gridDim,blockDim>>>(dVbo, Qinv, zhat, lss.LeastSquareImage(), dErr, within, c );
    return lss.FinalSystem();
}


}

#include "kangaroo.h"
#include "variational.h"
#include "launch_utils.h"
#include "Divergence.h"

using namespace std;

namespace Gpu
{

__global__ void KernTgvL1DenoisingDescentU(
    Image<float> imgu, Image<float2> imgp, Image<float> imgr, float alpha1, float tau
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgu.w && y < imgu.h ) {
        const float u = imgu(x,y);
        const float r = imgr(x,y);
        const float divp = DivA(imgp,x,y);
        imgu(x,y) = u - tau*(r - alpha1*divp);
    }
}

__global__ void KernTgvL1DenoisingDescentV(
    Image<float2> imgv, Image<float4> imgq, Image<float2> imgp, float alpha0, float alpha1, float tau
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgv.w && y < imgv.h ) {
        const float2 v = imgv(x,y);
        const float2 p = imgp(x,y);
        const float2 divq = DivA(imgq, x, y);
        imgv(x,y) = v - tau*(-alpha1*p - alpha0*divq);
    }
}

__global__ void KernTgvL1DenoisingAscentP(
    Image<float2> imgp, Image<float> imgu, Image<float2> imgv, float alpha1, float sigma
){
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgp.w && y < imgp.h ) {
        const float u = imgu(x,y);
        const float2 du = GradUFwd(imgu,u,x,y);
        const float2 v = imgv(x,y);
        const float2 p = imgp(x,y);

        imgp(x,y) = ProjectUnitBall( p + sigma * (alpha1 * (du-v)) );
    }
}

__global__ void KernTgvL1DenoisingAscentQ(
    Image<float4> imgq, Image<float2> imgv, float alpha0, float sigma
){
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgq.w && y < imgq.h ) {
        const float4 Epsv = Epsilon(imgv, x, y);
        const float4 q = imgq(x,y);
        imgq(x,y) = ProjectUnitBall(q + sigma * (alpha0 * Epsv) );
    }
}

__global__ void KernTgvL1DenoisingAscentR(
    Image<float> imgr, Image<float> imgu, Image<float> imgf, float delta, float sigma
){
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgr.w && y < imgr.h ) {
        const float r = imgr(x,y);
        const float u = imgu(x,y);
        const float f = imgf(x,y);
        imgr(x,y) = ProjectUnitBall( (r + sigma*(u-f) ) / (1.0f + sigma * delta) );
    }
}

__global__ void KernGradU(
    Image<float2> imgv, Image<float> imgu
){
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgv.w && y < imgv.h ) {
        const float u = imgu(x,y);
        const float2 du = GradUFwd(imgu,u,x,y);
        imgv(x,y) = du;
    }
}

void GradU(Image<float2> imgv, Image<float> imgu)
{
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);
    KernGradU<<<gridDim,blockDim>>>(imgv,imgu);
}

void TGV_L1_DenoisingIteration(
    Image<float> imgu, Image<float2> imgv,
    Image<float2> imgp, Image<float4> imgq, Image<float> imgr,
    Image<float> imgf,
    float alpha0, float alpha1,
    float sigma, float tau,
    float delta
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);

    KernTgvL1DenoisingAscentP<<<gridDim,blockDim>>>(imgp,imgu,imgv,alpha1,sigma);
    KernTgvL1DenoisingAscentQ<<<gridDim,blockDim>>>(imgq,imgv,alpha0,sigma);
    KernTgvL1DenoisingAscentR<<<gridDim,blockDim>>>(imgr,imgu,imgf,delta,sigma);
    KernTgvL1DenoisingDescentU<<<gridDim,blockDim>>>(imgu,imgp,imgr,alpha1,tau);
    KernTgvL1DenoisingDescentV<<<gridDim,blockDim>>>(imgv,imgq,imgp,alpha0,alpha1, tau);

}

}

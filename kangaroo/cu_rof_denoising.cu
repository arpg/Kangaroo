#include "kangaroo.h"
#include "variational.h"
#include "launch_utils.h"
#include "Divergence.h"

using namespace std;

namespace Gpu
{

//////////////////////////////////////////////////////
// Divergence, Div A
// da/dx + da/dy
//////////////////////////////////////////////////////

__global__ void KernDivergence(Image<float> divA, Image<float2> A)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < A.w && y < A.h ) {
        const float diva = DivA(A,x,y);
        divA(x,y) =  diva;
    }
}

void Divergence(Image<float> divA, Image<float2> A)
{
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, A);
    KernDivergence<<<gridDim,blockDim>>>(divA,A);
}

//////////////////////////////////////////////////////
// ROF p ascent L1
//////////////////////////////////////////////////////

__global__ void KernTVL1GradU_DualAscentP(
        Image<float2> p, const Image<float> u,
        float sigma
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < u.w && y < u.h ) {
        float2 du = make_float2(0,0);

        if(x < u.w-1 ) {
            du.x = u(x+1,y) - u(x,y);
        }

        if(y < u.h-1 ) {
            du.y = u(x,y+1) - u(x,y);
        }

        const float2 np = p(x,y) + sigma * du;
        const float mag_np = sqrt(np.x*np.x + np.y*np.y);
        const float reprojection = max(1.0f, mag_np);
        p(x,y) = np / reprojection;
    }
}

void TVL1GradU_DualAscentP(
        Image<float2> p, const Image<float> u,
        float sigma
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, u);
    KernTVL1GradU_DualAscentP<<<gridDim,blockDim>>>(p,u,sigma);
}

//////////////////////////////////////////////////////
// ROF p ascent Huber
//////////////////////////////////////////////////////

__global__ void KernHuberGradU_DualAscentP(
        Image<float2> p, const Image<float> u,
        float sigma, float alpha
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < u.w && y < u.h ) {
        float2 du = make_float2(0,0);

        if(x < u.w-1 ) {
            du.x = u(x+1,y) - u(x,y);
        }

        if(y < u.h-1 ) {
            du.y = u(x,y+1) - u(x,y);
        }

        const float2 np = (p(x,y) + sigma * du) / (1 + sigma*alpha);
        const float mag_np = sqrt(np.x*np.x + np.y*np.y);
        const float reprojection = max(1.0f, mag_np);
        p(x,y) = np / reprojection;
    }
}

void HuberGradU_DualAscentP(
        Image<float2> p, const Image<float> u,
        float sigma, float alpha
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, u);
    KernHuberGradU_DualAscentP<<<gridDim,blockDim>>>(p,u,sigma,alpha);
}

//////////////////////////////////////////////////////
// ROF u descent
//////////////////////////////////////////////////////

__global__ void KernL2_u_minus_g_PrimalDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg,
        float tau, float lambda
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgu.w && y < imgu.h ) {
        const float divp_np1 = DivA(imgp,x,y);

        const float g = imgg(x,y);
        const float u_n = imgu(x,y);
        const float u_np1 = (u_n + tau * (divp_np1 + lambda * g)) / (1.0f + tau*lambda);
        imgu(x,y) = u_np1;
    }
}

void L2_u_minus_g_PrimalDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg,
        float tau, float lambda
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);
    KernL2_u_minus_g_PrimalDescent<<<gridDim,blockDim>>>(imgu,imgp,imgg, tau, lambda);
}


} // namespace Gpu

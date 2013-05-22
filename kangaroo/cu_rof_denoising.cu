#include "variational.h"

#include "launch_utils.h"

using namespace std;

namespace roo
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
        Image<float2> imgp, const Image<float> imgu,
        float sigma
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgu.w && y < imgu.h ) {
        float2 du = make_float2(0,0);

        if(x < imgu.w-1 ) {
            du.x = imgu(x+1,y) - imgu(x,y);
        }

        if(y < imgu.h-1 ) {
            du.y = imgu(x,y+1) - imgu(x,y);
        }

        const float2 np = imgp(x,y) + sigma * du;
        const float mag_np = sqrt(np.x*np.x + np.y*np.y);
        const float reprojection = max(1.0f, mag_np);
        imgp(x,y) = np / reprojection;
    }
}

void TVL1GradU_DualAscentP(
        Image<float2> imgp, const Image<float> imgu,
        float sigma
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);
    KernTVL1GradU_DualAscentP<<<gridDim,blockDim>>>(imgp,imgu,sigma);
}

//////////////////////////////////////////////////////
// ROF p ascent Huber
//////////////////////////////////////////////////////

__global__ void KernHuberGradU_DualAscentP(
        Image<float2> imgp, const Image<float> imgu,
        float sigma, float alpha
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgu.w && y < imgu.h ) {
        const float u = imgu(x,y);
        float2 du = make_float2(0,0);

        if(x < imgu.w-1 ) {
            du.x = imgu(x+1,y) - u;
        }

        if(y < imgu.h-1 ) {
            du.y = imgu(x,y+1) - u;
        }

        const float2 np = (imgp(x,y) + sigma * du) / (1 + sigma*alpha);
        const float mag_np = sqrt(np.x*np.x + np.y*np.y);
        const float reprojection = max(1.0f, mag_np);
        imgp(x,y) = np / reprojection;
    }
}

void HuberGradU_DualAscentP(
        Image<float2> imgp, const Image<float> imgu,
        float sigma, float alpha
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);
    KernHuberGradU_DualAscentP<<<gridDim,blockDim>>>(imgp,imgu,sigma,alpha);
}

//////////////////////////////////////////////////////
// Grad-Weighted ROF p ascent Huber
//////////////////////////////////////////////////////

__global__ void KernWeightedHuberGradU_DualAscentP(
        Image<float2> imgp, const Image<float> imgu,
        const Image<float> imgw,
        float sigma, float alpha
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgu.w && y < imgu.h ) {
        const float u = imgu(x,y);
        const float2 du = GradUFwd(imgu,u,x,y);

        const float w = imgw(x,y);
        const float2 np = (imgp(x,y) + sigma * w * du) / (1 + sigma*alpha);
        const float mag_np = sqrt(np.x*np.x + np.y*np.y);
        const float reprojection = max(1.0f, mag_np);
        imgp(x,y) = np / reprojection;
    }
}

void WeightedHuberGradU_DualAscentP(
        Image<float2> imgp, const Image<float> imgu, const Image<float> imgw,
        float sigma, float alpha
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);
    KernWeightedHuberGradU_DualAscentP<<<gridDim,blockDim>>>(imgp,imgu,imgw,sigma,alpha);
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

//////////////////////////////////////////////////////
// ROF u descent
//////////////////////////////////////////////////////

__global__ void KernL2_u_minus_g_PrimalDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg,
        const Image<float> imglambdaweight,
        float tau, float lambda
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgu.w && y < imgu.h ) {
        lambda *= imglambdaweight(x,y);

        const float divp_np1 = DivA(imgp,x,y);

        const float g = imgg(x,y);
        const float u_n = imgu(x,y);
        const float u_np1 = (u_n + tau * (divp_np1 + lambda * g)) / (1.0f + tau*lambda);
        imgu(x,y) = u_np1;
    }
}

void L2_u_minus_g_PrimalDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg,
        const Image<float> imglambdaweight,
        float tau, float lambda
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);
    KernL2_u_minus_g_PrimalDescent<<<gridDim,blockDim>>>(imgu,imgp,imgg,imglambdaweight, tau, lambda);
}

//////////////////////////////////////////////////////
// Weighted ROF u descent
//////////////////////////////////////////////////////

__global__ void KernWeightedL2_u_minus_g_PrimalDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg, const Image<float> imgw,
        float tau, float lambda
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgu.w && y < imgu.h ) {
        const float divp_np1 = DivA(imgp,x,y);

        const float w = imgw(x,y);
        const float g = imgg(x,y);
        const float u_n = imgu(x,y);
        const float u_np1 = (u_n + tau * (w * divp_np1 + lambda * g)) / (1.0f + tau*lambda);
        imgu(x,y) = u_np1;
    }
}

void WeightedL2_u_minus_g_PrimalDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg, const Image<float> imgw,
        float tau, float lambda
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);
    KernWeightedL2_u_minus_g_PrimalDescent<<<gridDim,blockDim>>>(imgu,imgp,imgg,imgw, tau, lambda);
}


} // namespace roo

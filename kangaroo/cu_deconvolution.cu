#include "Image.h"
#include "variational.h"
#include "launch_utils.h"
#include "Divergence.h"

namespace Gpu {

//////////////////////////////////////////////////////
// Convolution p ascent
//////////////////////////////////////////////////////

__global__ void KernDeconvolutionDual_qAscent(
        Image<float> imgq, const Image<float> imgAu, const Image<float> imgg,
        float sigma_q, float lambda
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgq.w && y < imgq.h ) {
        const float q = imgq(x,y);
        const float Au = imgAu(x,y);
        const float g = imgg(x,y);
        const float q_np1 = ( q + sigma_q * (Au - g)) / (1.0f + sigma_q / lambda);
        imgq(x,y) = q_np1;
    }
}

void DeconvolutionDual_qAscent(
        Image<float> imgq, const Image<float> imgAu, const Image<float> imgg,
        float sigma_q, float lambda
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgq);
    KernDeconvolutionDual_qAscent<<<gridDim,blockDim>>>(imgq,imgAu,imgg,sigma_q,lambda);
}

//////////////////////////////////////////////////////
// Convolution u descent
//////////////////////////////////////////////////////

__global__ void KernDeconvolution_uDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgATq,
        float tau, float lambda
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgu.w && y < imgu.h ) {
        const float divp_np1 = DivA(imgp,x,y);

        const float ATq = imgATq(x,y);
        const float u_n = imgu(x,y);
        const float u_np1 = (u_n + tau * (divp_np1 - lambda * ATq)); // / (1.0f + tau*lambda);
        imgu(x,y) = u_np1;
    }
}

void Deconvolution_uDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgATq,
        float tau, float lambda
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);
    KernDeconvolution_uDescent<<<gridDim,blockDim>>>(imgu,imgp,imgATq, tau, lambda);
}

}

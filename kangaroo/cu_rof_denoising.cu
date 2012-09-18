#include "kangaroo.h"
#include "launch_utils.h"

namespace Gpu
{

__global__ void KernDenoisingRof_pAscent(
        Image<float2> p, Image<float> u,
        float sigma
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < u.w && y < u.h ) {
        float2 du = make_float2(0,0);

        if(x > 0 ) {
            du.x = u.GetBackwardDiffDx<float>(x,y);
        }

        if(y > 0 ) {
            du.y = u.GetBackwardDiffDy<float>(x,y);
        }

        float2 np = p(x,y) + sigma * du;
        p(x,y) = np;
    }
}

void DenoisingRof_pAscent(
        Image<float2> p, Image<float> u,
        float sigma, Image<unsigned char> scratch
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, u);
    KernDenoisingRof_pAscent<<<gridDim,blockDim>>>(p,u,sigma);
    const float sumabs = ImageL1<float, float2>(p, scratch);
    ElementwiseScaleBias<float2,float2,float2>(p, p, max(1.0f,sumabs), make_float2(0,0) );
}


} // namespace Gpu

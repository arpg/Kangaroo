#include "Image.h"
#include "launch_utils.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Make Index Buffer for rendering
//////////////////////////////////////////////////////

__global__ void KernGenerateTriangleStripIndexBuffer(Image<uint2> dIbo)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    uint2 idx;
    if(y%2) {
        const unsigned int pixIndex = (y+1)*dIbo.w - 1 - x;
        idx = make_uint2(pixIndex + dIbo.w, pixIndex);
    }else{
        const unsigned int pixIndex = y*dIbo.w + x;
        idx = make_uint2(pixIndex, pixIndex + dIbo.w);
    }
    dIbo(x,y) = idx;
}

void GenerateTriangleStripIndexBuffer( Image<uint2> dIbo)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dIbo);
    KernGenerateTriangleStripIndexBuffer<<<gridDim,blockDim>>>(dIbo);
}

}

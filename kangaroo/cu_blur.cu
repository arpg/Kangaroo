#include "kangaroo.h"
#include "launch_utils.h"

namespace Gpu {

template<typename TO, typename TI>
__global__ void KernBlurX(Image<TO> out, Image<TI> in)
{
    const unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x==0) {
        out(x,y) = (2*in(x,y) + in(x+1,y)) / 3.0f;
    }else if(x== in.w-1) {
        out(x,y) = (2*in(x,y) + in(x-1,y)) / 3.0f;
    }else{
        out(x,y) = (in(x-1,y) + 2*in(x,y) + in(x+1,y)) / 4.0f;
    }
}

template<typename TO, typename TI>
__global__ void KernBlurY(Image<TO> out, Image<TI> in)
{
    const unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y*blockDim.y + threadIdx.y;

    if(y==0) {
        out(x,y) = (2*in(x,y) + in(x,y+1)) / 3.0f;
    }else if(y== in.h-1) {
        out(x,y) = (2*in(x,y) + in(x,y-1)) / 3.0f;
    }else{
        out(x,y) = (in(x,y-1) + 2*in(x,y) + in(x,y+1)) / 4.0f;
    }
}

void Blur(Image<unsigned char> out, Image<unsigned char> in, Image<unsigned char> temp )
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, out, 16, 16);
    KernBlurX<unsigned char,unsigned char><<<gridDim,blockDim>>>(temp,in);
    KernBlurY<unsigned char,unsigned char><<<gridDim,blockDim>>>(out,temp);
}

}

#include "kangaroo.h"
#include "launch_utils.h"
#include "Image.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Image Addition
//////////////////////////////////////////////////////

template<typename Tout, typename Tin1, typename Tin2>
__global__ void KernAdd(Image<Tout> out, Image<Tin1> in1, Image<Tin2> in2)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(out.InBounds(x,y)) {
        out(x,y) = ConvertPixel<Tout,Tin1>(in1(x,y)) + ConvertPixel<Tout,Tin2>(in2(x,y));
    }
}

template<typename Tout, typename Tin1, typename Tin2>
void Add(Image<Tout> out, Image<Tin1> in1, Image<Tin2> in2 )
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim, out);
    KernAdd<Tout,Tin1,Tin2><<<gridDim,blockDim>>>(out,in1,in2);
}

//////////////////////////////////////////////////////
// Image Subtraction
//////////////////////////////////////////////////////

template<typename Tout, typename Tin1, typename Tin2, typename Tup>
__global__ void KernSubtractAdd(Image<Tout> out, Image<Tin1> in1, Image<Tin2> in2, Tup offset)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(out.InBounds(x,y)) {
        const Tup v1 = offset + ConvertPixel<Tup,Tin1>(in1(x,y));
        const Tup v2 = v1 - ConvertPixel<Tup,Tin2>(in2(x,y));
        out(x,y) = ConvertPixel<Tout,Tup>(v2);
    }
}

template<typename Tout, typename Tin1, typename Tin2, typename Tup>
void SubtractAdd(Image<Tout> out, Image<Tin1> in1, Image<Tin2> in2, Tup offset )
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim, out);
    KernSubtractAdd<Tout,Tin1,Tin2,Tup><<<gridDim,blockDim>>>(out,in1,in2,offset);
}

//////////////////////////////////////////////////////
// Instantiate Templates
//////////////////////////////////////////////////////

template void Add<unsigned char, unsigned char, unsigned char>(Image<unsigned char>, Image<unsigned char>, Image<unsigned char> in2);
template void SubtractAdd<unsigned char, unsigned char, unsigned char, int>(Image<unsigned char>, Image<unsigned char>, Image<unsigned char> in2, int );


}

#include "kangaroo.h"
#include "launch_utils.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Anaglyph: Join left / right images into anagly stereo
//////////////////////////////////////////////////////

__global__ void KernMakeAnaglythRedBlue(Image<uchar4> anaglyth, const Image<unsigned char> left, const Image<unsigned char> right, int shift)
{
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const unsigned char l = left(x,y);
    const unsigned char r = right.GetWithClampedRange(x-shift,y);

    anaglyth(x,y) = make_uchar4(l, 0, r, 255);
}

__global__ void KernMakeAnaglythColorCode(Image<uchar4> anaglyth, const Image<unsigned char> left, const Image<unsigned char> right, int shift)
{
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const unsigned char l = left(x,y);
    const unsigned char r = right.GetWithClampedRange(x-shift,y);

    const float3 mixleft  = make_float3(0x77, 0x77, 0x33);
    const float3 mixright = make_float3(0x00, 0x00, 0x99);
    const float3 out = (l * mixleft + r * mixright) / 255.0;

    anaglyth(x,y) = make_uchar4(out.x, out.y, out.z, 255);
}

void MakeAnaglyth(Image<uchar4> anaglyth, const Image<unsigned char> left, const Image<unsigned char> right, int shift)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, anaglyth.SubImage(shift,0, anaglyth.w-shift, anaglyth.h));
    KernMakeAnaglythColorCode<<<gridDim,blockDim>>>(anaglyth, left, right, shift);
}

}

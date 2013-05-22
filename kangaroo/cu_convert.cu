#include "cu_convert.h"

#include "launch_utils.h"
#include "pixel_convert.h"

namespace roo
{

//////////////////////////////////////////////////////
// Image Conversion
//////////////////////////////////////////////////////

template<typename To, typename Ti>
__global__
void KernConvertImage(Image<To> dOut, const Image<Ti> dIn)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(dOut.InBounds(x,y)) {
        dOut(x,y) = ConvertPixel<To,Ti>(dIn(x,y));
    }
}

template<typename To, typename Ti>
void ConvertImage(Image<To> dOut, const Image<Ti> dIn)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim, gridDim, dOut);
    KernConvertImage<<<gridDim,blockDim>>>(dOut,dIn);
}

// Explicit instantiation
template void ConvertImage<float,unsigned char>(Image<float>, const Image<unsigned char>);
template void ConvertImage<float,unsigned short>(Image<float>, const Image<unsigned short>);
template void ConvertImage<float,char>(Image<float>, const Image<char>);
template void ConvertImage<uchar4,uchar3>(Image<uchar4>, const Image<uchar3>);
template void ConvertImage<uchar3,uchar4>(Image<uchar3>, const Image<uchar4>);
template void ConvertImage<uchar3,unsigned char>(Image<uchar3>, const Image<unsigned char>);
template void ConvertImage<uchar4,unsigned char>(Image<uchar4>, const Image<unsigned char>);
template void ConvertImage<uchar4,float4>(Image<uchar4>, const Image<float4>);
template void ConvertImage<unsigned char, uchar3>(Image<unsigned char>, const Image<uchar3>);
template void ConvertImage<unsigned char, uchar4>(Image<unsigned char>, const Image<uchar4>);
template void ConvertImage<float4, float>(Image<float4>, const Image<float>);


} // namespace roo

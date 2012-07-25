#pragma once

#include <cuda_runtime.h>

namespace Gpu
{

//////////////////////////////////////////////////////
// Image Conversion
//////////////////////////////////////////////////////

template<typename To, typename Ti>
__host__ __device__
To ConvertPixel(Ti p)
{
    return p;
}

template<>
__host__ __device__
uchar4 ConvertPixel(unsigned char p)
{
    return make_uchar4(p,p,p,255);
}

template<>
__host__ __device__
uchar3 ConvertPixel(unsigned char p)
{
    return make_uchar3(p,p,p);
}

template<>
__host__ __device__
unsigned char ConvertPixel(uchar3 p)
{
    const unsigned sum = p.x + p.y + p.z;
    return sum / 3;
}

template<>
__host__ __device__
unsigned char ConvertPixel(uchar4 p)
{
    const unsigned sum = p.x + p.y + p.z;
    return sum / 3;
}

template<>
__host__ __device__
uchar4 ConvertPixel(uchar3 p)
{
    return make_uchar4(p.x,p.y,p.z,255);
}

template<>
__host__ __device__
uchar3 ConvertPixel(uchar4 p)
{
    return make_uchar3(p.x,p.y,p.z);
}

template<typename To, typename Ti>
__global__
void KernConvertImage(Image<To> dOut, const Image<Ti> dIn)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    dOut(x,y) = ConvertPixel<To,Ti>(dIn(x,y));
}

template<typename To, typename Ti>
void ConvertImage(Image<To> dOut, const Image<Ti> dIn)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dOut);
    KernConvertImage<<<gridDim,blockDim>>>(dOut,dIn);
}

// Explicit instantiation
template void ConvertImage<float,unsigned char>(Image<float>, const Image<unsigned char>);
template void ConvertImage<float,char>(Image<float>, const Image<char>);
template void ConvertImage<uchar4,uchar3>(Image<uchar4>, const Image<uchar3>);
template void ConvertImage<uchar3,uchar4>(Image<uchar3>, const Image<uchar4>);
template void ConvertImage<uchar3,unsigned char>(Image<uchar3>, const Image<unsigned char>);
template void ConvertImage<uchar4,unsigned char>(Image<uchar4>, const Image<unsigned char>);
template void ConvertImage<unsigned char, uchar3>(Image<unsigned char>, const Image<uchar3>);
template void ConvertImage<unsigned char, uchar4>(Image<unsigned char>, const Image<uchar4>);

}

#pragma once

#include <cuda_runtime.h>

namespace Gpu
{

template<typename To, typename Ti>
__host__ __device__ inline
To ConvertPixel(Ti p)
{
    return p;
}

template<>
__host__ __device__ inline
uchar4 ConvertPixel(unsigned char p)
{
    return make_uchar4(p,p,p,255);
}

template<>
__host__ __device__ inline
uchar3 ConvertPixel(unsigned char p)
{
    return make_uchar3(p,p,p);
}

template<>
__host__ __device__ inline
unsigned char ConvertPixel(uchar3 p)
{
    const unsigned sum = p.x + p.y + p.z;
    return sum / 3;
}

template<>
__host__ __device__ inline
unsigned char ConvertPixel(uchar4 p)
{
    const unsigned sum = p.x + p.y + p.z;
    return sum / 3;
}

template<>
__host__ __device__ inline
uchar4 ConvertPixel(uchar3 p)
{
    return make_uchar4(p.x,p.y,p.z,255);
}

template<>
__host__ __device__ inline
uchar4 ConvertPixel(float4 p)
{
    return make_uchar4(p.x*255,p.y*255,p.z*255,p.w*255);
}

template<>
__host__ __device__ inline
uchar3 ConvertPixel(uchar4 p)
{
    return make_uchar3(p.x,p.y,p.z);
}

template<>
__host__ __device__ inline
float4 ConvertPixel(float p)
{
    return make_float4(p,p,p,1.0);
}

template<>
__host__ __device__ inline
float3 ConvertPixel(uchar3 p)
{
    return make_float3(p.x,p.y,p.z);
}

template<>
__host__ __device__ inline
float4 ConvertPixel(uchar4 p)
{
    return make_float4(p.x,p.y,p.z,p.z);
}

}

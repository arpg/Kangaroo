#pragma once

#include <cuda_runtime.h>
#include "CUDA_SDK/cutil_math.h"

namespace Gpu
{

#define COLOR_SDF

struct __align__(8) SDF_t {
    inline __host__ __device__ SDF_t() {}
    inline __host__ __device__ SDF_t(float v) : val(v), w(1) {}
    inline __host__ __device__ SDF_t(float v, float w) : val(v), w(w) {}

    inline __host__ __device__ operator float() const {
        return val;
    }
    inline __host__ __device__ void Clamp(float minval, float maxval) {
        val = clamp(val, minval, maxval);
    }
    inline __host__ __device__ void LimitWeight(float max_weight) {
        w = fminf(w, max_weight);
    }
    inline __host__ __device__ void operator+=(const SDF_t& rhs)
    {
        if(rhs.w > 0) {
            val = (w * val + rhs.w * rhs.val);
            w += rhs.w;
            val /= w;
        }
    }

    float val;
    float w;
};

//struct __align__(8) SDF_t {
//    inline __host__ __device__ SDF_t() {}
//    inline __host__ __device__ SDF_t(float v) : val(v), w(1) {}
//    inline __host__ __device__ SDF_t(float v, float w) : val(v), w(w) {}
//    inline __host__ __device__ operator float() const {
//        return __half2float(val);
//    }
//    inline __host__ __device__ void Clamp(float minval, float maxval) {
//        val = __float2half_rn( clamp(__half2float(val), minval, maxval) );
//    }
//    inline __host__ __device__ void LimitWeight(float max_weight) {
//        w = __float2half_rn(fminf( __half2float(w), max_weight));
//    }
//    inline __host__ __device__ void operator+=(const SDF_t& rhs)
//    {
//        if(__half2float(rhs.w) > 0) {
//            val = __float2half_rn(__half2float(w) * __half2float(val) + __half2float(rhs.w) * __half2float(rhs.val) );
//            w = __float2half_rn(__half2float(w) + __half2float(rhs.w));
//            val = __float2half_rn(__half2float(val) / __half2float(w));
//        }
//    }

//    unsigned short w;
//    unsigned short val;
//};

inline __host__ __device__ SDF_t operator+(const SDF_t& lhs, const SDF_t& rhs)
{
    SDF_t res = lhs;
    res += rhs;
    return res;
}

}

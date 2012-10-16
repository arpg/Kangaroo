#pragma once

#include <cuda_runtime.h>

namespace Gpu
{

struct SDF_t {
    inline __host__ __device__ SDF_t() {}
    inline __host__ __device__ SDF_t(float v) : val(v), w(1) {}
    inline __host__ __device__ SDF_t(float v, int w) : val(v), w(w) {}
    inline __host__ __device__ operator float() const {
        return val;
    }    
    float w;
    float val;
};

inline __host__ __device__ SDF_t operator+(SDF_t lhs, SDF_t rhs)
{
    SDF_t res;
    res.w = lhs.w + rhs.w;
    res.val = (lhs.w * lhs.val + rhs.w * rhs.val) / res.w;
    return res;
}


}

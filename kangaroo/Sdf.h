#pragma once

#include <cuda_runtime.h>
#include "CUDA_SDK/cutil_math.h"

namespace Gpu
{

struct SDF_t {
    inline __host__ __device__ SDF_t() {}
    inline __host__ __device__ SDF_t(float v) : val(v), w(1) {}
    inline __host__ __device__ SDF_t(float v, float w) : val(v), w(w) {}
    inline __host__ __device__ operator float() const {
        return w * (val / w);
//        return val;
    }    
    inline __host__ __device__ void Clamp(float minval, float maxval) {
        val = clamp(val, minval, maxval);
    }
    inline __host__ __device__ void LimitWeight(float max_weight) {
        w = fminf(w, max_weight);
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

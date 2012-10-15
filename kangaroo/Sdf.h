#pragma once

#include <cuda_runtime.h>

namespace Gpu
{

struct SDF_t {
    inline __host__ __device__ SDF_t() {}
    inline __host__ __device__ SDF_t(float v) : val(v), n(1) {}
    inline __host__ __device__ SDF_t(float v, int n) : val(v), n(n) {}
    inline __host__ __device__ operator float() const {
        return val / n;
    }
    float val;
    int n;
};

}

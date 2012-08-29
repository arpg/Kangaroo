#pragma once

#include <cuda_runtime.h>

namespace Gpu
{

template<typename T>
struct InvalidValue;

template<>
struct InvalidValue<float> {
    inline __host__ __device__ static float Value() {
        return 0.0f/0.0f;
    }

    inline __host__ __device__ static bool IsValid(float val) {
        return isfinite(val);
    }
};

template<>
struct InvalidValue<char> {
    inline __host__ __device__ static float Value() {
        return 0;
    }

    inline __host__ __device__ static bool IsValid(unsigned char val) {
        return !val;
    }
};


template<>
struct InvalidValue<unsigned char> {
    inline __host__ __device__ static float Value() {
        return 0;
    }

    inline __host__ __device__ static bool IsValid(unsigned char val) {
        return !val;
    }
};

template<>
struct InvalidValue<int> {
    inline __host__ __device__ static float Value() {
        return -1;
    }

    inline __host__ __device__ static bool IsValid(int val) {
        return val >= 0;
    }
};

} // namespace

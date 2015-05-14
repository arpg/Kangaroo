#include <cuda_runtime.h>

namespace roo {

__host__ __device__ inline
float LSReweightSq(float /*r*/, float /*c*/) {
    return 1;
}

__host__ __device__ inline
float LSReweightL1(float r, float /*c*/) {
    const float absr = fabs(r);
    return 1.0f / absr;
}

__host__ __device__ inline
float LSReweightHuber(float r, float c) {
    const float absr = fabs(r);
    return (absr <= c ) ? 1.0f : c / absr;
}

__host__ __device__ inline
float LSReweightTukey(float r, float c) {
    const float absr = fabs(r);
    const float roc = r / c;
    const float omroc2 = 1.0f - roc*roc;
    return (absr <= c ) ? omroc2*omroc2 : 0.0f;
}

__host__ __device__ inline
float LSReweightCauchy(float r, float c) {
    const float roc = r / c;
    return 1.0f / (1.0f + roc*roc);
}

}

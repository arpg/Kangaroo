#include "CUDA_SDK/cutil_math.h"
#include "CudaImage.h"

namespace Gpu
{

template<typename T, size_t Size>
struct Array
{
    __host__ __device__
    inline T& operator[](size_t i) {
        return arr[i];
    }

    __host__ __device__
    inline const T& operator[](size_t i) const {
        return arr[i];
    }

    T arr[Size];
};

//////////////////////////////////////////////////////

void CreateMatlabLookupTable(Image<float2> lookup,
    float fu, float fv, float u0, float v0, float k1, float k2
);

void CreateMatlabLookupTable(Image<float2> lookup,
    float fu, float fv, float u0, float v0, float k1, float k2, Array<float,9> H_no
);

//////////////////////////////////////////////////////

void Warp(
    Image<uchar1> out, const Image<uchar1> in, const Image<float2> lookup
);

//////////////////////////////////////////////////////

void MakeAnaglyth(
    Image<uchar4> anaglyth,
    const Image<uchar1> left, const Image<uchar1> right
);

}

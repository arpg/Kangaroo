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
    Image<unsigned char> out, const Image<unsigned char> in, const Image<float2> lookup
);

//////////////////////////////////////////////////////

void DenseStereo(
    Image<float> dDisp, Image<unsigned char> dCamLeft, Image<unsigned char> dCamRight, int maxDisp
);

//////////////////////////////////////////////////////

void DenseStereoSubpixelRefine(
    Image<float> dDisp, Image<unsigned char> dCamLeft, Image<unsigned char> dCamRight, int maxDisp
);

//////////////////////////////////////////////////////

void BilateralFilter(
    Image<float> dOut, Image<float> dIn, float gs, float gr, uint size
);

//////////////////////////////////////////////////////

void MakeAnaglyth(
    Image<uchar4> anaglyth,
    const Image<unsigned char> left, const Image<unsigned char> right
);

}

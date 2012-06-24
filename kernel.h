#include "CUDA_SDK/cutil_math.h"
#include "CudaImage.h"

namespace Gpu
{

struct __align__(8) LookupWeights
{
    uint ix;
    uchar1 w[4];
};

void CreateMatlabLookupTable(
    Image<LookupWeights>& lookup,
    float fu, float fv, float u0, float v0, float k1, float k2
);

void MakeAnaglyth(Image<uchar4> anaglyth, const Image<uchar1> left, const Image<uchar1> right);

}

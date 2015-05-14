#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Mat.h>
#include <kangaroo/Image.h>

namespace roo
{

KANGAROO_EXPORT
void CreateMatlabLookupTable(Image<float2> lookup,
    float fu, float fv, float u0, float v0, float k1, float k2
);

KANGAROO_EXPORT
void CreateMatlabLookupTable(Image<float2> lookup,
    float fu, float fv, float u0, float v0, float k1, float k2, Mat<float,9> H_no
);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void Warp(
    Image<unsigned char> out, const Image<unsigned char> in, const Image<float2> lookup
);

}

#pragma once

#include "Mat.h"
#include "Image.h"

namespace roo
{

LeastSquaresSystem<float,3> ManhattenLineCost(
    Image<float4> out, Image<float4> out2, const Image<unsigned char> in,
    Mat<float,3,3> Rhat, float fu, float fv, float u0, float v0,
    float cut, float scale, float min_grad,
    Image<unsigned char> dWorkspace
);

}

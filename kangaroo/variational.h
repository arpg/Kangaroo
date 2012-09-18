#pragma once

#include "kangaroo.h"

namespace Gpu {

void Divergence(Image<float> divA, Image<float2> A);

void DenoisingRof_pAscent(
        Image<float2> p, Image<float> u,
        float sigma, Image<unsigned char> scratch
);

void DenoisingRof_uDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg,
        float tau, float lambda
);

}

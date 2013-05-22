#pragma once

#include "Image.h"

namespace roo
{

void DeconvolutionDual_qAscent(
        Image<float> q, const Image<float> Au, const Image<float> g,
        float sigma_q, float lambda
);

void Deconvolution_uDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgATq,
        float tau, float lambda
);

}

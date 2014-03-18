#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>

namespace roo
{

KANGAROO_EXPORT
void DeconvolutionDual_qAscent(
        Image<float> q, const Image<float> Au, const Image<float> g,
        float sigma_q, float lambda
);

KANGAROO_EXPORT
void Deconvolution_uDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgATq,
        float tau, float lambda
);

}

#pragma once

#include <kangaroo/platform.h>

namespace roo
{

KANGAROO_EXPORT
void GradU(Image<float2> imgv, Image<float> imgu);

KANGAROO_EXPORT
void TGV_L1_DenoisingIteration(
    Image<float> imgu, Image<float2> imgv,
    Image<float2> imgp, Image<float4> imgq, Image<float> imgr,
    Image<float> imgf,
    float alpha0, float alpha1,
    float sigma, float tau,
    float delta
);

}

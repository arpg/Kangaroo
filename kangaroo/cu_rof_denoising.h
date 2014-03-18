#pragma once

#include <kangaroo/platform.h>

namespace roo
{

KANGAROO_EXPORT
void Divergence(Image<float> divA, Image<float2> A);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void TVL1GradU_DualAscentP(
        Image<float2> p, Image<float> u,
        float sigma
);

KANGAROO_EXPORT
void HuberGradU_DualAscentP(
        Image<float2> p, const Image<float> u,
        float sigma, float alpha
);

KANGAROO_EXPORT
void WeightedHuberGradU_DualAscentP(
        Image<float2> imgp, const Image<float> imgu, const Image<float> imgw,
        float sigma, float alpha
);

KANGAROO_EXPORT
void L2_u_minus_g_PrimalDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg,
        float tau, float lambda
);

KANGAROO_EXPORT
void L2_u_minus_g_PrimalDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg,
        const Image<float> imglambdaweight,
        float tau, float lambda
);

KANGAROO_EXPORT
void WeightedL2_u_minus_g_PrimalDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg, const Image<float> imgw,
        float tau, float lambda
);

}

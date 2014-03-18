#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>

namespace roo
{

template<typename To, typename Ti>
KANGAROO_EXPORT
void BilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, uint size
);

template<typename To, typename Ti>
KANGAROO_EXPORT
void BilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, uint size, Ti minval
);

template<typename To, typename Ti, typename Ti2>
KANGAROO_EXPORT
void BilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, const Image<Ti2> dImg, float gs, float gr, float gc, uint size
);

}

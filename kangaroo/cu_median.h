#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>

namespace roo
{

KANGAROO_EXPORT
void MedianFilter3x3(
    Image<float> dOut, Image<float> dIn
);

KANGAROO_EXPORT
void MedianFilter5x5(
    Image<float> dOut, Image<float> dIn
);

KANGAROO_EXPORT
void MedianFilterRejectNegative5x5(
    Image<float> dOut, Image<float> dIn, int maxbad = 100
);

KANGAROO_EXPORT
void MedianFilterRejectNegative7x7(
    Image<float> dOut, Image<float> dIn, int maxbad
);

KANGAROO_EXPORT
void MedianFilterRejectNegative9x9(
    Image<float> dOut, Image<float> dIn, int maxbad
);

}

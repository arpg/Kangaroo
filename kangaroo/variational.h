#pragma once

#include "kangaroo.h"

namespace Gpu {

void DenoisingRof_pAscent(
        Image<float2> p, Image<float> u,
        float sigma, Image<unsigned char> scratch
);

}

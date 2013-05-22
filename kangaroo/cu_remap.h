#pragma once

#include "Image.h"

namespace roo
{

void Remap(Image<float4> out, const Image<float> img, const Image<float> score, float min, float max);

}

#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>

namespace roo
{

KANGAROO_EXPORT
void Remap(Image<float4> out, const Image<float> img, const Image<float> score, float min, float max);

}

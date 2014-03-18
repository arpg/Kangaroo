#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>

namespace roo
{

KANGAROO_EXPORT
void NormalsFromVbo(Image<float4> dN, const Image<float4> dV);

}

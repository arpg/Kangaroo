#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>

namespace roo
{

template<typename To, typename Ti>
KANGAROO_EXPORT
void ConvertImage(Image<To> dOut, const Image<Ti> dIn);

}

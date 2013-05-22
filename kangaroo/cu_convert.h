#pragma once

#include "Image.h"

namespace roo
{

template<typename To, typename Ti>
void ConvertImage(Image<To> dOut, const Image<Ti> dIn);

}

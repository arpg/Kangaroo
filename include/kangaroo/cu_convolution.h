#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>

namespace roo
{

template<typename OT, typename IT, typename KT, typename ACC>
KANGAROO_EXPORT
void Convolution(
    Image<OT> out,  Image<IT> in,  Image<KT> kern, int kx, int ky
);

}

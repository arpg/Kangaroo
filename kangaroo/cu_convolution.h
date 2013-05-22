#pragma once

#include "Image.h"

namespace roo
{

template<typename OT, typename IT, typename KT, typename ACC>
void Convolution(
    Image<OT> out,  Image<IT> in,  Image<KT> kern, int kx, int ky
);

}

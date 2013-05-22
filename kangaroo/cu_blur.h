#pragma once

#include "Image.h"

namespace roo
{

void Blur(Image<unsigned char> out, Image<unsigned char> in, Image<unsigned char> temp );

inline void Blur(Image<unsigned char> in_out, Image<unsigned char> temp )
{
    Blur(in_out,in_out,temp);
}

template<typename Tout, typename Tin, unsigned MAXRAD, unsigned MAXIMGDIM>
void GaussianBlur(Image<Tout> out, Image<Tin> in, Image<Tout> temp, float sigma);

}

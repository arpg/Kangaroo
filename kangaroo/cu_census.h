#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>
#include <kangaroo/Volume.h>

namespace roo
{

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void Census(Image<unsigned long> census, Image<unsigned char> img);

KANGAROO_EXPORT
void Census(Image<ulong2> census, Image<unsigned char> img);

KANGAROO_EXPORT
void Census(Image<ulong4> census, Image<unsigned char> img);


KANGAROO_EXPORT
void Census(Image<unsigned long> census, Image<float> img);

KANGAROO_EXPORT
void Census(Image<ulong2> census, Image<float> img);

KANGAROO_EXPORT
void Census(Image<ulong4> census, Image<float> img);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void CensusStereo(Image<char> disp, Image<unsigned long> left, Image<unsigned long> right, int maxDisp);

template<typename Tvol, typename T>
KANGAROO_EXPORT
void CensusStereoVolume(Volume<Tvol> vol, Image<T> left, Image<T> right, int maxDisp, float sd);

}

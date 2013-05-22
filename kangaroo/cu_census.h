#pragma once

#include "Image.h"
#include "Volume.h"

namespace roo
{

//////////////////////////////////////////////////////

void Census(Image<unsigned long> census, Image<unsigned char> img);
void Census(Image<ulong2> census, Image<unsigned char> img);
void Census(Image<ulong4> census, Image<unsigned char> img);

void Census(Image<unsigned long> census, Image<float> img);
void Census(Image<ulong2> census, Image<float> img);
void Census(Image<ulong4> census, Image<float> img);

//////////////////////////////////////////////////////

void CensusStereo(Image<char> disp, Image<unsigned long> left, Image<unsigned long> right, int maxDisp);

template<typename Tvol, typename T>
void CensusStereoVolume(Volume<Tvol> vol, Image<T> left, Image<T> right, int maxDisp, float sd);

}

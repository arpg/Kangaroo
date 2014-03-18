#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>
#include <kangaroo/Volume.h>

namespace roo
{

template<typename TH, typename TC, typename Timg>
KANGAROO_EXPORT
void SemiGlobalMatching(Volume<TH> volH, Volume<TC> volC, Image<Timg> left, int maxDisp, float P1, float P2, bool dohoriz, bool dovert, bool doreverse);

}

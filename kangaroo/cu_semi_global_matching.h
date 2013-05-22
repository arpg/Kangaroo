#pragma once

#include "Image.h"
#include "Volume.h"

namespace roo
{

template<typename TH, typename TC, typename Timg>
void SemiGlobalMatching(Volume<TH> volH, Volume<TC> volC, Image<Timg> left, int maxDisp, float P1, float P2, bool dohoriz, bool dovert, bool doreverse);

}

#pragma once

#include "Image.h"

namespace roo
{

void SegmentTest(
    Image<unsigned char> out, const Image<unsigned char> img, unsigned char threshold, unsigned char min_segment_len
);

void HarrisScore(
    Image<float> out, const Image<unsigned char> img, float lambda = 0.04
);

void NonMaximalSuppression(Image<unsigned char> out, Image<float> scores, int rad, float threshold);

}

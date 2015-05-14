#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>

namespace roo
{

KANGAROO_EXPORT
void SegmentTest(
    Image<unsigned char> out, const Image<unsigned char> img, unsigned char threshold, unsigned char min_segment_len
);

KANGAROO_EXPORT
void HarrisScore(
    Image<float> out, const Image<unsigned char> img, float lambda = 0.04
);

KANGAROO_EXPORT
void NonMaximalSuppression(Image<unsigned char> out, Image<float> scores, int rad, float threshold);

}

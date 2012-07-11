#pragma once

namespace Gpu
{

//////////////////////////////////////////////////////
// Patch Scores
//////////////////////////////////////////////////////

template<typename T, int size>
__device__ inline
float Sum(
    Image<T> img, int x, int y
) {
    float sum = 0;
    for(int r=-size; r <=size; ++r ) {
        for(int c=-size; c <=size; ++c ) {
            sum += img.GetWithClampedRange(x+c,y+r);
        }
    }
    return sum;
}

// Mean Absolute Difference
template<typename To, typename T, int rad>
__device__ inline
float MADScore(
    Image<T> img1, int x1, int y1,
    Image<T> img2, int x2, int y2
) {
    const int w = 2*rad+1;
    To sum_abs_diff = 0;

    for(int r=-rad; r <=rad; ++r ) {
        for(int c=-rad; c <=rad; ++c ) {
            T i1 = img1.GetWithClampedRange(x1+c,y1+r);
            T i2 = img2.GetWithClampedRange(x2+c,y2+r);
            sum_abs_diff += abs(i1 - i2);
        }
    }

    return sum_abs_diff / (w*w);
}

// Sum Square Normalised Difference
template<typename To, typename T, int rad>
__device__ inline
To SSNDScore(
    Image<T> img1, int x1, int y1,
    Image<T> img2, int x2, int y2
) {
//    // Straightforward approach
//    const int w = 2*rad+1;
//    const float m1 = Sum<T,rad>(img1,x1,y1) / (w*w);
//    const float m2 = Sum<T,rad>(img2,x2,y2) / (w*w);

//    float sum_abs_diff = 0;
//    for(int r=-rad; r <=rad; ++r ) {
//        for(int c=-rad; c <=rad; ++c ) {
//            float i1 = img1.GetWithClampedRange(x1+c,y1+r) - m1;
//            float i2 = img2.GetWithClampedRange(x2+c,y2+r) - m2;
//            sum_abs_diff += abs(i1 - i2);
//        }
//    }
//    return sum_abs_diff;

    To sxi = 0;
    To sxi2 = 0;
    To syi = 0;
    To syi2 = 0;
    To sxiyi = 0;

    const int w = 2*rad+1;
    const int n = w*w;

    for(int r=-rad; r <=rad; ++r ) {
        for(int c=-rad; c <=rad; ++c ) {
            To xi = img1.GetWithClampedRange(x1+c,y1+r);
            To yi = img2.GetWithClampedRange(x2+c,y2+r);
            sxi += xi;
            syi += yi;
            sxi2 += xi*xi;
            syi2 += yi*yi;
            sxiyi += xi*yi;
        }
    }

    const To mx = (float)sxi / (float)n;
    const To my = (float)syi / (float)n;

    const To score = 0
            + sxi2 - 2*mx*sxi + n*mx*mx
            + 2*(-sxiyi + my*sxi + mx*syi - n*mx*my)
            + syi2 - 2*my*syi + n*my*my;
    return score;
}

}

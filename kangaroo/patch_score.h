#pragma once

#include "Image.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Image Access
//////////////////////////////////////////////////////

struct ImgAccessRaw
{
    template<typename T>
    __host__ __device__ inline static
    T Get(const Image<T>& img, int x, int y) {
        return img(x,y);
    }
};

struct ImgAccessClamped
{
    template<typename T>
    __host__ __device__ inline static
    T Get(const Image<T>& img, int x, int y) {
        return img.GetWithClampedRange(x,y);
    }
};

template<typename Tinterp>
struct ImgAccessBilinear
{
    template<typename T>
    __host__ __device__ inline static
    T Get(const Image<T>& img, float x, float y) {
        return img.template GetBilinear<Tinterp>(x,y);
    }
};

template<typename Tinterp>
struct ImgAccessBilinearClamped
{
    template<typename T>
    __host__ __device__ inline static
    T Get(const Image<T>& img, float x, float y) {
        if(x<0) x=0;
        if(x > img.w-1) x = img.w-1;
        if(y<0) y=0;
        if(y > img.h-1) y = img.h-1;
        return img.template GetBilinear<Tinterp>(x,y);
    }
};

//////////////////////////////////////////////////////
// Patch Scores
//////////////////////////////////////////////////////

template<typename To, typename T, int rad, typename ImgAccess>
__host__ __device__ inline
To Sum(
    Image<T> img, int x, int y
) {
    To sum = 0;
    for(int r=-rad; r <=rad; ++r ) {
        for(int c=-rad; c <=rad; ++c ) {
            sum += ImgAccess::Get(img,x+c,y+r);
        }
    }
    return sum;
}

// Mean Absolute Difference
template<typename To, typename ImgAccess = ImgAccessRaw >
struct SinglePixelSqPatchScore {
    static const int width = 1;
    static const int height = 1;
    static const int area = width*height;

    template<typename T>
    __host__ __device__ inline static
    To Score(
        Image<T> img1, int x1, int y1,
        Image<T> img2, int x2, int y2
    ) {
        const T i1 = ImgAccess::Get(img1,x1,y1);
        const T i2 = ImgAccess::Get(img2,x2,y2);
        const To diff = (To)(i1 - i2);
        return diff*diff;
    }
};

// Sum Absolute Difference
template<typename To, int rad=1, typename ImgAccess = ImgAccessRaw >
struct SADPatchScore {
    static const int width = 2*rad+1;
    static const int height = 2*rad+1;
    static const int area = width*height;

    template<typename T>
    __host__ __device__ inline static
    To Score(
        Image<T> img1, int x1, int y1,
        Image<T> img2, int x2, int y2
    ) {
        To sum_abs_diff = 0;

        for(int r=-rad; r <=rad; ++r ) {
            for(int c=-rad; c <=rad; ++c ) {
                const T i1 = ImgAccess::Get(img1,x1+c,y1+r);
                const T i2 = ImgAccess::Get(img2,x2+c,y2+r);
                sum_abs_diff += abs(i1 - i2);
            }
        }

        return sum_abs_diff;
    }
};

// Sum Absolute Difference
template<typename To, int rad=1, typename ImgAccess = ImgAccessRaw >
struct SSDPatchScore {
    static const int width = 2*rad+1;
    static const int height = 2*rad+1;
    static const int area = width*height;

    template<typename T>
    __host__ __device__ inline static
    To Score(
        Image<T> img1, int x1, int y1,
        Image<T> img2, int x2, int y2
    ) {
        To sum_sq_diff = 0;

        for(int r=-rad; r <=rad; ++r ) {
            for(int c=-rad; c <=rad; ++c ) {
                const T i1 = ImgAccess::Get(img1,x1+c,y1+r);
                const T i2 = ImgAccess::Get(img2,x2+c,y2+r);
                const To diff = i1 - i2;
                sum_sq_diff += diff*diff;
            }
        }

        return sum_sq_diff;
    }
};

// Sum Square Normalised Difference
template<typename To, int rad=1, typename ImgAccess = ImgAccessRaw >
struct SSNDPatchScore {
    static const int width = 2*rad+1;
    static const int height = 2*rad+1;
    static const int area = width*height;

    template<typename T>
    __host__ __device__ inline static
    To Score(
        Image<T> img1, int x1, int y1,
        Image<T> img2, int x2, int y2
    ) {
        To sxi = 0;
        To sxi2 = 0;
        To syi = 0;
        To syi2 = 0;
        To sxiyi = 0;

        const int w = 2*rad+1;
        const int n = w*w;

        for(int r=-rad; r <=rad; ++r ) {
            #pragma unroll
            for(int c=-rad; c <=rad; ++c ) {
                const To xi = ImgAccess::Get(img1,x1+c,y1+r);
                const To yi = ImgAccess::Get(img2,x2+c,y2+r);
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
};

// Sum Square Normalised Difference for line
template<typename To, int rad=1, typename ImgAccess = ImgAccessRaw >
struct SSNDLineScore
{
    static const int width = 2*rad+1;
    static const int height = 1;
    static const int area = width*height;

    template<typename T>
    __host__ __device__ inline static
    To Score(
        Image<T> img1, int x1, int y1,
        Image<T> img2, int x2, int y2
    ) {
        To sxi = 0;
        To sxi2 = 0;
        To syi = 0;
        To syi2 = 0;
        To sxiyi = 0;

        const int w = 2*rad+1;
        const int n = w*w;

        #pragma unroll
        for(int c=-rad; c <=rad; ++c ) {
            const To xi = ImgAccess::Get(img1,x1+c,y1);
            const To yi = ImgAccess::Get(img2,x2+c,y2);
            sxi += xi;
            syi += yi;
            sxi2 += xi*xi;
            syi2 += yi*yi;
            sxiyi += xi*yi;
        }

        const To mx = (float)sxi / (float)n;
        const To my = (float)syi / (float)n;

        const To score = 0
                + sxi2 - 2*mx*sxi + n*mx*mx
                + 2*(-sxiyi + my*sxi + mx*syi - n*mx*my)
                + syi2 - 2*my*syi + n*my*my;
        return score;
    }
};

// Sum Absolute Normalised Difference
template<typename To, int rad=1, typename ImgAccess = ImgAccessRaw >
struct SANDPatchScore {
    static const int width = 2*rad+1;
    static const int height = 2*rad+1;
    static const int area = width*height;

    template<typename T>
    __host__ __device__ inline static
    To Score(
        Image<T> img1, int x1, int y1,
        Image<T> img2, int x2, int y2
    ) {
        To sum_abs_diff = 0;

        To sum1 = 0;
        To sum2 = 0;

        for(int r=-rad; r <=rad; ++r ) {
            for(int c=-rad; c <=rad; ++c ) {
                const T i1 = ImgAccess::Get(img1,x1+c,y1+r);
                const T i2 = ImgAccess::Get(img2,x2+c,y2+r);
                sum1 += i1;
                sum2 += i2;
            }
        }

        const To mean1 = sum1 / area;
        const To mean2 = sum2 / area;

        for(int r=-rad; r <=rad; ++r ) {
            for(int c=-rad; c <=rad; ++c ) {
                const T i1 = ImgAccess::Get(img1,x1+c,y1+r);
                const T i2 = ImgAccess::Get(img2,x2+c,y2+r);
                sum_abs_diff += abs( (i1-mean1) - (i2-mean2) );
            }
        }

        return sum_abs_diff;
    }
};

}

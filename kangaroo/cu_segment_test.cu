#include "kangaroo.h"
#include "launch_utils.h"

namespace Gpu {

//////////////////////////////////////////////////////
// Segment test
//////////////////////////////////////////////////////

template<typename Tup>
__host__ __device__
inline void PixelTest(Tup p, Tup q, unsigned short bit, unsigned short& dark, unsigned short& light, Tup threshold)
{
    if( p + threshold < q ) {
        // light
        light |= bit;
    }else if( q < p - threshold ) {
        // dark
        dark |= bit;
    }
}

template<typename Tout, typename T, typename Tup>
__global__ void KernSegmentTest(
    Image<Tout> out, const Image<T> img, Tup threshold, unsigned char min_segment_len = 9
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // TODO: Use apron of 3 pixels to accelerate global memory reads.
    const int rad = 3;

    if(x < img.w && y < img.h ) {
        const Tup p = img(x,y);

        unsigned short light = 0;
        unsigned short dark = 0;

        Tout response = 0;

        if( img.InBounds(x,y,rad) ) {
            // Perform all tests
            PixelTest<Tup>(p, img(x-1,y-3), 1<< 0, dark, light, threshold);
            PixelTest<Tup>(p, img(x+0,y-3), 1<< 1, dark, light, threshold);
            PixelTest<Tup>(p, img(x+1,y-3), 1<< 2, dark, light, threshold);
            PixelTest<Tup>(p, img(x-2,y-2), 1<<15, dark, light, threshold);
            PixelTest<Tup>(p, img(x+2,y-2), 1<< 3, dark, light, threshold);
            PixelTest<Tup>(p, img(x-3,y-1), 1<<14, dark, light, threshold);
            PixelTest<Tup>(p, img(x+3,y-1), 1<< 4, dark, light, threshold);
            PixelTest<Tup>(p, img(x-3,y+0), 1<<13, dark, light, threshold);
            PixelTest<Tup>(p, img(x+3,y+0), 1<< 5, dark, light, threshold);
            PixelTest<Tup>(p, img(x-3,y+1), 1<<12, dark, light, threshold);
            PixelTest<Tup>(p, img(x+3,y+1), 1<< 6, dark, light, threshold);
            PixelTest<Tup>(p, img(x-2,y+2), 1<<11, dark, light, threshold);
            PixelTest<Tup>(p, img(x+2,y+2), 1<< 7, dark, light, threshold);
            PixelTest<Tup>(p, img(x-1,y+3), 1<<10, dark, light, threshold);
            PixelTest<Tup>(p, img(x+0,y+3), 1<< 9, dark, light, threshold);
            PixelTest<Tup>(p, img(x+1,y+3), 1<< 8, dark, light, threshold);

            // Bitwise string of opposite values in circle.
            const unsigned short opplight = (light >> 8) | (light << 8);
            const unsigned short oppdark  = (dark >> 8) | (light << 8);

            // min_segment_len = 10 for FAST 12 (12 segment out of 16)
            const bool corner = (__popc(light&opplight) >= min_segment_len) || (__popc(dark&oppdark) >= min_segment_len);

//            // (Still not) FAST 9 (9 segment out of 16)
//            const bool corner =
//                    ( ((light | (light^opplight)) == 0xffff)  && __popc(light) >= min_segment_len) ||
//                    ( (( dark | ( dark^ oppdark)) == 0xffff)  && __popc(dark ) >= min_segment_len);

            response = corner ? 255 : 0;

        }

        out(x,y) = response;
    }
}

void SegmentTest(
    Image<unsigned char> out, const Image<unsigned char> img, unsigned char threshold, unsigned char min_segment_len
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, img);
    KernSegmentTest<unsigned char, unsigned char, int><<<gridDim,blockDim>>>(out, img, threshold, min_segment_len);
}

//////////////////////////////////////////////////////
// Harris Corner score
//////////////////////////////////////////////////////

template<typename Tout, typename T, typename Tup>
__global__ void KernHarrisScore(
    Image<Tout> out, const Image<T> img, float lambda = 0.04
) {
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    const int rad = 1;

    if( x < img.w && y < img.h ) {
        float score = 0;

        if(rad < x && x < img.w-rad && rad < y && y < img.h - rad ) {
            float sum_Ixx = 0;
            float sum_Iyy = 0;
            float sum_Ixy = 0;

            for(int sy=-rad; sy<=rad; ++sy) {
                for(int sx=-rad; sx<=rad; ++sx) {
                    Mat<float,1,2> dI = img.template GetCentralDiff<float>(x+sx,y+sy);
                    sum_Ixx += dI(0) * dI(0);
                    sum_Iyy += dI(1) * dI(1);
                    sum_Ixy += dI(0) * dI(1);
                }
            }

            const int area = (2*rad+1)*(2*rad+1);
            sum_Ixx /= area;
            sum_Iyy /= area;
            sum_Ixy /= area;

            const float det = sum_Ixx * sum_Iyy - sum_Ixy * sum_Ixy;
            const float trace = sum_Ixx + sum_Iyy;
            score = det - lambda * trace*trace;
        }

        out(x,y) = score;
    }
}

void HarrisScore(
    Image<float> out, const Image<unsigned char> img, float lambda
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, img);
    KernHarrisScore<float, unsigned char, int><<<gridDim,blockDim>>>(out, img, lambda);
}

//////////////////////////////////////////////////////
// Non-maximal suppression
//////////////////////////////////////////////////////

template<typename Tout, typename T>
__global__ void KernNonMaximalSuppression(
    Image<Tout> out, const Image<T> img, int rad, float threshold
) {
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < img.w && y < img.h ) {
        float val = 0;

        if(rad < x && x < img.w-rad && rad < y && y < img.h - rad ) {
            const T p = img(x,y);
            T score = p;

            for(int sy=-rad; sy<=rad; ++sy) {
                for(int sx=-rad; sx<=rad; ++sx) {
                    const T q = img(x+sx, y+sy);
                    if(q >= p && !(sx==0 && sy==0) ) {
                        score = 0;
                    }
                }
            }

            val = score > threshold ? 255 : 0;
        }

        out(x,y) = val;
    }
}

void NonMaximalSuppression(Image<unsigned char> out, Image<float> scores, int rad, float threshold)
{
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, out);
    KernNonMaximalSuppression<unsigned char,float><<<gridDim,blockDim>>>(out, scores, rad, threshold);
}

}

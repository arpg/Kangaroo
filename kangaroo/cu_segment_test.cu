#include "kangaroo.h"
#include "launch_utils.h"

namespace Gpu {

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

template<typename T, typename Tout, typename Tup>
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

}

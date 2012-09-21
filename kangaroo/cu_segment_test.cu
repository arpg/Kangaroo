#include "kangaroo.h"
#include "launch_utils.h"

namespace Gpu {

template<typename T>
__host__ __device__
inline void PixelTest(T p, T q, int pos, unsigned int& dark, unsigned int& light, T threshold)
{
    if( p + threshold < q ) {
        // light
        light |= 1 << pos;
    }else if( q < p - threshold ) {
        // dark
        dark |= 1 << pos;
    }
}

template<typename T, typename Tout>
__global__ void KernSegmentTest(
    Image<Tout> out, const Image<T> img, T threshold
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    const int rad = 3;

    if(x < img.w && y < img.h ) {
        const T p = img(x,y);

        unsigned int light = 0;
        unsigned int dark = 0;

        Tout response = 0;

        if( img.InBounds(x,y,rad) ) {
            // Perform all tests
            PixelTest<T>(p, img(x-1,y-3),  0, dark, light, threshold);
            PixelTest<T>(p, img(x+0,y-3),  1, dark, light, threshold);
            PixelTest<T>(p, img(x+1,y-3),  2, dark, light, threshold);
            PixelTest<T>(p, img(x-2,y-2), 15, dark, light, threshold);
            PixelTest<T>(p, img(x+2,y-2),  3, dark, light, threshold);
            PixelTest<T>(p, img(x-3,y-1), 14, dark, light, threshold);
            PixelTest<T>(p, img(x+3,y-1),  4, dark, light, threshold);
            PixelTest<T>(p, img(x-3,y+0), 13, dark, light, threshold);
            PixelTest<T>(p, img(x+3,y+0),  5, dark, light, threshold);
            PixelTest<T>(p, img(x-3,y+1), 12, dark, light, threshold);
            PixelTest<T>(p, img(x+3,y+1),  6, dark, light, threshold);
            PixelTest<T>(p, img(x-2,y+2), 11, dark, light, threshold);
            PixelTest<T>(p, img(x+2,y+2),  7, dark, light, threshold);
            PixelTest<T>(p, img(x-1,y+3), 10, dark, light, threshold);
            PixelTest<T>(p, img(x+0,y+3),  9, dark, light, threshold);
            PixelTest<T>(p, img(x+1,y+3),  8, dark, light, threshold);

            // Bitwise string of opposite values in circle.
            const unsigned int opplight = (light >> 8) | (light << 8);
            const unsigned int oppdark  = (dark >> 8) | (light << 8);

            // FAST 12 (12 segment out of 16)
            const bool corner = (__popc(light&opplight) >= 16) || (__popc(dark&oppdark) >= 16);

            response = corner ? 255 : 0;

        }

        out(x,y) = response;
    }
}

void SegmentTest(
    Image<unsigned char> out, const Image<unsigned char> img, unsigned char threshold
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, img);
    KernSegmentTest<unsigned char, unsigned char><<<gridDim,blockDim>>>(out, img, threshold);
}

}

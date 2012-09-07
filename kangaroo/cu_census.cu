#include "kangaroo.h"
#include "Image.h"
#include "hamming_distance.h"
#include "launch_utils.h"
#include "InvalidValue.h"

namespace Gpu
{

const int MaxImageWidth = 1024;

//////////////////////////////////////////////////////
// Census transform, 9x7 window
//////////////////////////////////////////////////////

template<typename Tout, typename Tin>
__global__ void KernCensus9x7(Image<Tout> census, Image<Tin> img)
{
    const int WRAD = 4;
    const int HRAD = 3;

    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if( img.InBounds(x,y) ) {
        const Tin p = img(x,y);

        Tout out = 0;
        Tout bit = 1;

        for(int r=-HRAD; r <= HRAD; ++r) {
#pragma unroll
            for(int c=-WRAD; c <= WRAD; ++c ) {
                const Tin q = img.GetWithClampedRange(x+c,y+r);
                if( q < p ) {
                    out |= bit;
                }
                bit = bit << 1;
            }
        }

        census(x,y) = out;
    }
}

//////////////////////////////////////////////////////
// Census transform, 11x11 window
//////////////////////////////////////////////////////

template<typename Tin>
__global__ void KernCensus11x11(Image<ulong2> census, Image<Tin> img)
{
    const int WRAD = 5;
    const int HRAD = 5;

    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if( img.InBounds(x,y) ) {
        const Tin p = img(x,y);

        ulong2 out = make_ulong2(0,0);
        unsigned long bit = 1;

        for(int r=-HRAD; r < 0; ++r) {
#pragma unroll
            for(int c=-WRAD; c <= WRAD; ++c ) {
                const Tin q = img.GetWithClampedRange(x+c,y+r);
                if( q < p ) {
                    out.x |= bit;
                }
                bit = bit << 1;
            }
        }

#pragma unroll
        for(int c=-WRAD; c <= 0; ++c ) {
            const Tin q = img.GetWithClampedRange(x+c,y);
            if( q < p ) {
                out.x |= bit;
            }
            bit = bit << 1;
        }

        bit = 1;
#pragma unroll
        for(int c=1; c <= WRAD; ++c ) {
            const Tin q = img.GetWithClampedRange(x+c,y);
            if( q < p ) {
                out.y |= bit;
            }
            bit = bit << 1;
        }

        for(int r=1; r <= HRAD; ++r) {
#pragma unroll
            for(int c=-WRAD; c <= WRAD; ++c ) {
                const Tin q = img.GetWithClampedRange(x+c,y+r);
                if( q < p ) {
                    out.y |= bit;
                }
                bit = bit << 1;
            }
        }

        census(x,y) = out;
    }
}

//////////////////////////////////////////////////////
// Census transform, 16x16 window
//////////////////////////////////////////////////////

template<typename Tin>
__global__ void KernCensus16x16(Image<ulong4> census, Image<Tin> img)
{
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if( img.InBounds(x,y) ) {
        const Tin p = img(x,y);

        ulong4 out = make_ulong4(0,0,0,0);
        unsigned long bit = 1;

        for(int r=-8; r < -4; ++r) {
#pragma unroll
            for(int c=-4; c < 4; ++c ) {
                const Tin q = img.GetWithClampedRange(x+c,y+r);
                if( q < p ) {
                    out.x |= bit;
                }
                bit = bit << 1;
            }
        }

        bit = 1;
        for(int r=-4; r < 0; ++r) {
#pragma unroll
            for(int c=-4; c < 4; ++c ) {
                const Tin q = img.GetWithClampedRange(x+c,y+r);
                if( q < p ) {
                    out.y |= bit;
                }
                bit = bit << 1;
            }
        }

        bit = 1;
        for(int r=0; r < 4; ++r) {
#pragma unroll
            for(int c=-4; c < 4; ++c ) {
                const Tin q = img.GetWithClampedRange(x+c,y+r);
                if( q < p ) {
                    out.z |= bit;
                }
                bit = bit << 1;
            }
        }

        bit = 1;
        for(int r=4; r < 8; ++r) {
#pragma unroll
            for(int c=-4; c < 4; ++c ) {
                const Tin q = img.GetWithClampedRange(x+c,y+r);
                if( q < p ) {
                    out.w |= bit;
                }
                bit = bit << 1;
            }
        }
        census(x,y) = out;
    }

}


void Census(Image<unsigned long> census, Image<unsigned char> img)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim,img);
    KernCensus9x7<unsigned long, unsigned char><<<gridDim,blockDim>>>(census,img);
}

void Census(Image<ulong2> census, Image<unsigned char> img)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim,img);
    KernCensus11x11<unsigned char><<<gridDim,blockDim>>>(census,img);
}

void Census(Image<ulong4> census, Image<unsigned char> img)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim,img);
    KernCensus16x16<unsigned char><<<gridDim,blockDim>>>(census,img);
}

//////////////////////////////////////////////////////
// Census Stereo
//////////////////////////////////////////////////////

template<typename T>
__global__ void KernCensusStereo(Image<char> disp, Image<T> left, Image<T> right, int maxDispVal)
{
    const int x = threadIdx.x;
    const int y = blockIdx.y;

    __shared__ T cache_r[MaxImageWidth];
    cache_r[x] = right(x,y);
    __syncthreads();

    const T p = left(x,y);

    unsigned bestScore = 0xFFFFF;
    int bestDisp = InvalidValue<char>::Value();

    int minDisp = min(maxDispVal, 0);
    int maxDisp = max(0, maxDispVal);
    minDisp = max(minDisp, x - ((int)left.w-1));
    maxDisp = min(maxDisp, x);

    for(int d=minDisp; d< maxDisp; ++d)
    {
        const int xd = x-d;
        const T q = cache_r[xd]; //right(xd,y);
        const unsigned score = HammingDistance(p,q);

        if(score < bestScore) {
            bestScore = score;
            bestDisp = d;
        }
    }

    disp(x,y) = bestDisp;
}

void CensusStereo(Image<char> disp, Image<unsigned long> left, Image<unsigned long> right, int maxDisp)
{
    dim3 blockDim(disp.w, 1);
    dim3 gridDim(1, disp.h);
    KernCensusStereo<unsigned long><<<gridDim,blockDim>>>(disp,left,right,maxDisp);
}

//////////////////////////////////////////////////////
// Build Census Cost volume
//////////////////////////////////////////////////////

template<typename Tvol, typename T>
__global__ void KernCensusStereoVolume(Volume<Tvol> vol, Image<T> left, Image<T> right, int maxDispVal)
{
    const int x = threadIdx.x;
    const int y = blockIdx.y;

    __shared__ T cache_r[MaxImageWidth];
    cache_r[x] = right(x,y);
    __syncthreads();

    const T p = left(x,y);

    const int maxDisp = min(maxDispVal, x+1);

    for(int d=0; d< maxDisp; ++d)
    {
        const int xd = x-d;
        const T q = cache_r[xd]; //right(xd,y);
        const Tvol score = HammingDistance(p,q);
        vol(x,y,d) = score;
    }
}


void CensusStereoVolume(Volume<unsigned short> vol, Image<unsigned long> left, Image<unsigned long> right, int maxDisp)
{
    dim3 blockDim(left.w, 1);
    dim3 gridDim(1, left.h);
    KernCensusStereoVolume<unsigned short, unsigned long><<<gridDim,blockDim>>>(vol,left,right,maxDisp);
}

void CensusStereoVolume(Volume<unsigned short> vol, Image<ulong2> left, Image<ulong2> right, int maxDisp)
{
    dim3 blockDim(left.w, 1);
    dim3 gridDim(1, left.h);
    KernCensusStereoVolume<unsigned short, ulong2><<<gridDim,blockDim>>>(vol,left,right,maxDisp);
}

void CensusStereoVolume(Volume<unsigned short> vol, Image<ulong4> left, Image<ulong4> right, int maxDisp)
{
    dim3 blockDim(left.w, 1);
    dim3 gridDim(1, left.h);
    KernCensusStereoVolume<unsigned short, ulong4><<<gridDim,blockDim>>>(vol,left,right,maxDisp);
}

}

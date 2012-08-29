#pragma once

#include "Image.h"
#include "InvalidValue.h"

namespace Gpu
{

// This class MUST be instantiated as __shared__
template<typename T, unsigned Top, unsigned Bottom, unsigned Left, unsigned Right, unsigned BLOCKW, unsigned BLOCKH>
struct ImageApron
{
public:
    const static unsigned Width = BLOCKW + Left + Right;
    const static unsigned Height = BLOCKH + Top + Bottom;
    const static unsigned Elements = Width * Height;
    const static unsigned ThreadsInBlock = BLOCKW * BLOCKH;

    // Conservative: how do we work this out properly? Compile time ceil()?
    const static unsigned Loads = 1 + Elements / (BLOCKW*BLOCKH);

    //! Cache image for block, with block thread (0,0) having pixel (x,y)
    //! You likely want to set:  x = (blockIdx.x*blockDim.x)
    //!                          y = (blockIdx.y*blockDim.y)
    inline __host__ __device__
    void CacheImage(const Image<T>& img, unsigned int x, unsigned y)
    {
        const uint tid = threadIdx.y*blockDim.x + threadIdx.x;

#pragma unroll
        for(int l=0; l<Loads; ++l) {
            const uint elid = l*ThreadsInBlock + tid;
            const uint imgx = x - Left + (elid % Width);
            const uint imgy = y - Top  + (elid / Width);
            if( 0 <= imgx && imgx < img.w && 0 <= imgy && imgy < img.h ) {
                cache[elid] = img.Get(imgx,imgy);
            }else{
                cache[elid] = InvalidValue<T>::Value();
            }
        }

        // Sync since this cache has been loaded by many threads
        __syncthreads();
    }

    inline __host__ __device__
    T GetRaw(uint x, uint y)
    {
        return cache[y*Width + x];
    }

    //! x and y are relative to block 0,0.
    inline __host__ __device__
    T GetRelBlock(uint x, uint y)
    {
        return GetRaw(x+Left, y+Top);
    }

    //! x and y are relative to current thread in block.
    inline __host__ __device__
    T GetRelThread(uint x, uint y)
    {
        return GetRelBlock(x+threadIdx.x, y+threadIdx.y);
    }

protected:
    T cache[Elements];
};

}

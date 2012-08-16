#include "kangaroo.h"
#include "launch_utils.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Experimental and dirty Bilateral filer
//////////////////////////////////////////////////////

template<typename To, typename Ti>
__global__ void KernRobustBilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, float go, int size
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const Ti p = dIn(x,y);
    float sum = 0;
    float sumw = 0;

    for(int r = -size; r <= size; ++r ) {
        for(int c = -size; c <= size; ++c ) {
            const Ti q = dIn.GetWithClampedRange(x+c, y+r);
            const float sd2 = r*r + c*c;
            const float id = p-q;
            const float id2 = id*id;
            const float sw = __expf(-(sd2) / (2 * gs * gs));
            const float iw = __expf(-(id2) / (2 * gr * gr));
            const float w = sw*iw;
            sumw += w;
            sum += w * q;
//            sumw += 1;
//            sum += q;
        }
    }

    dOut(x,y) = (To)(sum / sumw);

    syncthreads();

    const float r = (sumw-1) / sumw;

    if( r < go ) {
//        // Downweight pixel as outlier
//        sum -= p;
//        sumw -=1;

        // Downweight each pixel
        for(int r = -size; r <= size; ++r ) {
            for(int c = -size; c <= size; ++c ) {
                const To q = dOut.GetWithClampedRange(x+c, y+r);
                const float sd2 = r*r + c*c;
                const float id = p-q;
                const float id2 = id*id;
                const float sw = __expf(-(sd2) / (2 * gs * gs));
                const float iw = __expf(-(id2) / (2 * gr * gr));
                const float w = sw*iw;
                const float rq = (sumw-w) / sumw;
                if(rq < go) {
                    sum -= w*q;
                    sumw -= w;
                }
            }
        }
    }

    dOut(x,y) = (To)(sum / sumw);
//    dOut(x,y) = r;
}

void RobustBilateralFilter(
    Image<float> dOut, const Image<unsigned char> dIn, float gs, float gr, float go, uint size
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut);
    KernRobustBilateralFilter<float,unsigned char><<<gridDim,blockDim>>>(dOut, dIn, gs, gr, go, size);
}

//////////////////////////////////////////////////////
// Quick and dirty Bilateral filer
//////////////////////////////////////////////////////

template<typename To, typename Ti>
__global__ void KernBilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, int size
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const Ti p = dIn(x,y);
    float sum = 0;
    float sumw = 0;

    for(int r = -size; r <= size; ++r ) {
        for(int c = -size; c <= size; ++c ) {
            const Ti q = dIn.GetWithClampedRange(x+c, y+r);
            const float sd2 = r*r + c*c;
            const float id = p-q;
            const float id2 = id*id;
            const float sw = __expf(-(sd2) / (2 * gs * gs));
            const float iw = __expf(-(id2) / (2 * gr * gr));
            const float w = sw*iw;
            sumw += w;
            sum += w * q;
//            sumw += 1;
//            sum += q;
        }
    }

    dOut(x,y) = (To)(sum / sumw);
}

void BilateralFilter(
    Image<float> dOut, const Image<float> dIn, float gs, float gr, uint size
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut);
    KernBilateralFilter<float,float><<<gridDim,blockDim>>>(dOut, dIn, gs, gr, size);
}

void BilateralFilter(
    Image<float> dOut, const Image<unsigned char> dIn, float gs, float gr, uint size
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut);
    KernBilateralFilter<float,unsigned char><<<gridDim,blockDim>>>(dOut, dIn, gs, gr, size);
}

void BilateralFilter(
    Image<float> dOut, const Image<unsigned short> dIn, float gs, float gr, uint size
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut);
    KernBilateralFilter<float,unsigned short><<<gridDim,blockDim>>>(dOut, dIn, gs, gr, size);
}

}

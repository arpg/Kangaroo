#include "kangaroo.h"
#include "launch_utils.h"

namespace Gpu
{

/////////////////////////////////////////////////////
// Bilateral Filter (Spatial and intensity weights)
//////////////////////////////////////////////////////

template<typename To, typename Ti>
__global__ void KernBilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, int size
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if( dOut.InBounds(x,y)) {
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
            }
        }

        dOut(x,y) = (To)(sum / sumw);
    }
}

template<typename To, typename Ti>
void BilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, uint size
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim, dOut);
    KernBilateralFilter<To,Ti><<<gridDim,blockDim>>>(dOut, dIn, gs, gr, size);
}

template void BilateralFilter(Image<float>, const Image<float>, float, float, uint);
template void BilateralFilter(Image<float>, const Image<unsigned char>, float, float, uint);

/////////////////////////////////////////////////////
// Bilateral Filter (Spatial, intensity and colour (external) weights)
//////////////////////////////////////////////////////

template<typename To, typename Ti, typename Ti2>
__global__ void KernBilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, const Image<Ti2> dImg, float gs, float gr, float gc, int size
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if( dOut.InBounds(x,y)) {
        const float p = dIn(x,y);
        const float pc = dImg(x,y);
        float sum = 0;
        float sumw = 0;

        for(int r = -size; r <= size; ++r ) {
            for(int c = -size; c <= size; ++c ) {
                const float q = dIn.GetWithClampedRange(x+c, y+r);
                const float qc = dImg.GetWithClampedRange(x+c, y+r);
                const float rd = p-q;
                const float cd = pc-qc;
                const float sd2 = r*r + c*c;
                const float rd2 = rd*rd;
                const float cd2 = cd*cd;
                const float sw = __expf(-(sd2) / (2 * gs * gs));
                const float rw = __expf(-(rd2) / (2 * gr * gr));
                const float cw = __expf(-(cd2) / (2 * gc * gc));
                const float w = sw*rw*cw;
                sumw += w;
                sum += w * q;
            }
        }

        dOut(x,y) = sumw == 0 ? p : (To)(sum / sumw);
    }
}

template<typename To, typename Ti, typename Ti2>
void BilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, const Image<Ti2> dImg, float gs, float gr, float gc, uint size
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim, dOut);
    KernBilateralFilter<To,Ti><<<gridDim,blockDim>>>(dOut, dIn, dImg, gs, gr, gc, size);
}

template void BilateralFilter(Image<float>, const Image<float>, const Image<unsigned char>, float, float, float, uint);


}

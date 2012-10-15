#include "Image.h"
#include "launch_utils.h"

namespace Gpu
{

// Crudest possible CUDA Convolution implementation
template<typename OT, typename IT, typename KT, typename ACC>
__global__ void KernConvolution(
    Image<OT> out,  Image<IT> in,  Image<KT> kern, int kx, int ky
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < out.w && y < out.h) {
        ACC pixsum = 0;
        ACC kernsum = 0;
        for(int r=0; r< kern.h; ++r) {
            for(int c=0; c< kern.w; ++c) {
                const int sx = x - kx + c;
                const int sy = y - ky + r;
//                if( 0 <= sx && sx < out.w && 0 <= sy && sy < out.h ) {
//                    const KT kv = kern(c,r);
//                    kernsum += kv;
//                    pixsum += in(sx,sy) * kv;
//                }
                const KT kv = kern(c,r);
                kernsum += kv;

                pixsum += in.GetConditionNeumann(abs(sx),sy) * kv;
            }
        }
        out(x,y) = pixsum / kernsum;
    }
}

template<typename OT, typename IT, typename KT, typename ACC>
void Convolution(
    Image<OT> out,  Image<IT> in,  Image<KT> kern, int kx, int ky
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, out);
    KernConvolution<OT,IT,KT,ACC><<<gridDim,blockDim>>>(out,in,kern,kx,ky);
}

//////////////////////////////////////////////////////
// Instantiate templates
//////////////////////////////////////////////////////

template void Convolution<float,float,float,float>(Image<float> out,  Image<float> in,  Image<float> kern, int kx, int ky);
template void Convolution<float,unsigned char,unsigned char,float>(Image<float> out,  Image<unsigned char> in,  Image<unsigned char> kern, int kx, int ky);

}


#include "kangaroo.h"
#include "launch_utils.h"
#include "sampling.h"

namespace Gpu
{

__global__ void  resample_kernal(
    float4* out, int ostride, int ow, int oh,
    float4* in,  int istride, int iw, int ih,
    int resample_type
) {
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int index = y*ostride + x;

    const float xf = ((x+0.5) / (float)ow) * (float)iw;
    const float yf = ((y+0.5) / (float)oh) * (float)ih;

    if( 1.5 <= xf && xf < iw-2.5 && 1.5 <= yf && yf < ih-2.5 ) {
      if( resample_type == 1 ) {
        out[index] = bilinear_continuous<float4,float4>(in,istride,xf,yf);
      }else if( resample_type == 2 ) {
        out[index] = bicubic_continuous<float4,float4>(in,istride,xf,yf);
      }else if( resample_type == 3 ) {
        out[index] = catrom_continuous<float4,float4>(in,istride,xf,yf);
      }else{
        out[index] = nearestneighbour_continuous<float4,float4>(in,istride,xf,yf);
      }
    }
}


void Resample(
    float4* out, int ostride, int ow, int oh,
    float4* in,  int istride, int iw, int ih,
    int resample_type
) {
  dim3 blockdim(boost::math::gcd<unsigned>(ow,16), boost::math::gcd<unsigned>(oh,16), 1);
  dim3 griddim( ow / blockdim.x, oh / blockdim.y);
  resample_kernal<<<griddim,blockdim>>>(out,ostride,ow,oh,in,istride,iw,ih, resample_type);
}

//////////////////////////////////////////////////////
// Downsampling
//////////////////////////////////////////////////////

template<typename To, typename UpType, typename Ti>
__global__ void KernBoxHalf( Image<To> out, const Image<Ti> in )
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    const Ti* tl = &in(2*x,2*y);
    const Ti* bl = &in(2*x,2*y+1);

    out(x,y) = (To)((UpType)(*tl + *(tl+1) + *bl + *(bl+1)) / 4.0f);
}

template<typename To, typename UpType, typename Ti>
void BoxHalf( Image<To> out, const Image<Ti> in)
{
    dim3 blockDim;
    dim3 gridDim;
    InitDimFromOutputImage(blockDim,gridDim, out, 16, 16);
    KernBoxHalf<To,UpType,Ti><<<gridDim,blockDim>>>(out,in);
}

// Instantiate
template void BoxHalf<unsigned char,unsigned int,unsigned char>(Image<unsigned char>, const Image<unsigned char>);

}

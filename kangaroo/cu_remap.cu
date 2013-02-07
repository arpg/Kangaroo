#include "MatUtils.h"
#include "Image.h"
#include "launch_utils.h"

namespace Gpu
{

// h [0,360)
// s [0,1]
// v [0,1]
inline __host__ __device__
float4 hsv2rgb( double hue, double s, double v )
{
  const double h = hue / 60.0;
  const int i = floor(h);
  const double f = (i%2 == 0) ? 1-(h-i) : h-i;
  const double m = v * (1-s);
  const double n = v * (1-s*f);
  switch(i)
  {
  case 0:  return make_float4(v,n,m,1);
  case 1:  return make_float4(n,v,m,1);
  case 2:  return make_float4(m,v,n,1);
  case 3:  return make_float4(m,n,v,1);
  case 4:  return make_float4(n,m,v,1);
  default: return make_float4(v,m,n,1);
  }
}

__global__ void KernRemap(Image<float4> out, const Image<float> img, const Image<float> score, float in_min, float in_max)
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    if( u < out.w && v < out.h ) {
        // Create value in interval [0,1]
        const float iv = (score(u,v) - in_min) / (in_max - in_min);
        const float ov = iv;
        float ci = img(u,v);
        if(ci == 0.0f) ci = 1.0;
        const float4 ci4 = make_float4(ci,ci,ci,1);
        const float mix = 2*abs(0.5 - ov);
        const float4 cm4 = hsv2rgb(360*ov, 1.0, 1.0 );
        out(u,v) = (1-mix)*ci4 + mix*cm4;
    }
}

void Remap(Image<float4> out, const Image<float> img, const Image<float> score, float in_min, float in_max)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim, gridDim, out);
    KernRemap<<<gridDim,blockDim>>>(out, img, score, in_min, in_max);
    GpuCheckErrors();
    
}

}

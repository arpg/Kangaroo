#include "kernel.h"

#include "CUDA_SDK/cutil_math.h"
#include <boost/math/common_factor.hpp>

using namespace std;
using namespace boost::math;

//////////////////////////////////////////////////////
// Additions to cutil_math.h
//////////////////////////////////////////////////////

inline __host__ __device__ float3 operator*(float b, uchar3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ float3 operator*(uchar3 a, float b)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ float1 operator*(float b, uchar1 a)
{
    return make_float1(b * a.x);
}

inline __host__ __device__ float1 operator*(uchar1 a, float b)
{
    return make_float1(b * a.x);
}

//////////////////////////////////////////////////////
// Sampling
//////////////////////////////////////////////////////

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
//    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
float w1(float a)
{
//    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

__device__ float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

// filter 4 values using cubic splines
template<typename R, typename T>
__device__
R cubicFilter(float x, T c0, T c1, T c2, T c3)
{
    R r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

// Catmull-Rom interpolation

__host__ __device__
float catrom_w0(float a)
{
    //return -0.5f*a + a*a - 0.5f*a*a*a;
    return a*(-0.5f + a*(1.0f - 0.5f*a));
}

__host__ __device__
float catrom_w1(float a)
{
    //return 1.0f - 2.5f*a*a + 1.5f*a*a*a;
    return 1.0f + a*a*(-2.5f + 1.5f*a);
}

__host__ __device__
float catrom_w2(float a)
{
    //return 0.5f*a + 2.0f*a*a - 1.5f*a*a*a;
    return a*(0.5f + a*(2.0f - 1.5f*a));
}

__host__ __device__
float catrom_w3(float a)
{
    //return -0.5f*a*a + 0.5f*a*a*a;
    return a*a*(-0.5f + 0.5f*a);
}

template<typename R, typename T>
__device__
R catRomFilter(float x, T c0, T c1, T c2, T c3)
{
    R r;
    r = c0 * catrom_w0(x);
    r += c1 * catrom_w1(x);
    r += c2 * catrom_w2(x);
    r += c3 * catrom_w3(x);
    return r;
}

template<typename R, typename T>
__device__ R nearestneighbour(const T* img, int stride, float x, float y)
{
  const int xi = floor(x);
  const int yi = floor(y);
  return img[xi + stride*yi];
}

template<typename R, typename T>
__device__ R bilinear(const T* img, int stride, float x, float y)
{
  const float px = x - 0.5f;
  const float py = y - 0.5f;

//  if( 0.0 <= px && px < w-1.0 && 0.0 <= py && py < h-1.0 ) {
    const float ix = floorf(px);
    const float iy = floorf(py);
    const float fx = px - ix;
    const float fy = py - iy;
    const int idx = (int)ix + (int)iy*stride;

    return lerp(
      lerp( img[idx], img[idx+1], fx ),
      lerp( img[idx+stride], img[idx+stride+1], fx ),
      fy
    );
//  }else{
//    return nearestneighbour(img,stride,w,h,x,y);
//  }
}

template<typename R, typename T>
__device__ R bicubic(const T* img, int stride, float x, float y)
{
  const float px = x-0.5f;
  const float py = y-0.5f;

//  if( 1.0 <= px && px < w-2.0 && 1.0 <= py && py < h-2.0 ) {
    const int ix = floor(px);
    const int iy = floor(py);
    const float fx = px - ix;
    const float fy = py - iy;
    const int idx = ((int)ix) + ((int)iy)*stride;

    return cubicFilter<R,R>(
          fy,
          cubicFilter<R,T>(fx, img[idx-stride-1], img[idx-stride], img[idx-stride+1], img[idx-stride+2]),
          cubicFilter<R,T>(fx, img[idx-1], img[idx], img[idx+1], img[idx+2]),
          cubicFilter<R,T>(fx, img[idx+stride-1], img[idx+stride], img[idx+stride+1], img[idx+stride+2]),
          cubicFilter<R,T>(fx, img[idx+2*stride-1], img[idx+2*stride], img[idx+2*stride+1], img[idx+2*stride+2])
    );
//  }else{
//    return nearestneighbour(img,stride,w,h,x,y);
//  }
}

template<typename R, typename T>
__device__ R catrom(const T* img, uint stride, float x, float y)
{
  const float px = x-0.5f;
  const float py = y-0.5f;

//  if( 1.0 <= px && px < w-2.0 && 1.0 <= py && py < h-2.0 ) {
    const int ix = floor(px);
    const int iy = floor(py);
    const float fx = px - ix;
    const float fy = py - iy;
    const uint idx = ((uint)ix) + ((uint)iy)*stride;
    const uint stride2 = 2 *stride;

    return catRomFilter<R,R>(
          fy,
          catRomFilter<R,T>(fx, img[idx-stride-1], img[idx-stride], img[idx-stride+1], img[idx-stride+2]),
          catRomFilter<R,T>(fx, img[idx-1], img[idx], img[idx+1], img[idx+2]),
          catRomFilter<R,T>(fx, img[idx+stride-1], img[idx+stride], img[idx+stride+1], img[idx+stride+2]),
          catRomFilter<R,T>(fx, img[idx+stride2-1], img[idx+stride2], img[idx+stride2+1], img[idx+stride2+2])
    );
//  }else{
//    return nearestneighbour<R,T>(img,stride,x,y);
//  }
}

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
        out[index] = bilinear<float4,float4>(in,istride,xf,yf);
      }else if( resample_type == 2 ) {
        out[index] = bicubic<float4,float4>(in,istride,xf,yf);
      }else if( resample_type == 3 ) {
        out[index] = catrom<float4,float4>(in,istride,xf,yf);
      }else{
        out[index] = nearestneighbour<float4,float4>(in,istride,xf,yf);
      }
    }
}


void resample(
    float4* out, int ostride, int ow, int oh,
    float4* in,  int istride, int iw, int ih,
    int resample_type
) {
  dim3 blockdim(boost::math::gcd<unsigned>(ow,16), boost::math::gcd<unsigned>(oh,16), 1);
  dim3 griddim( ow / blockdim.x, oh / blockdim.y);
  resample_kernal<<<griddim,blockdim>>>(out,ostride,ow,oh,in,istride,iw,ih, resample_type);
}

//////////////////////////////////////////////////////
// Image warping
//////////////////////////////////////////////////////

namespace Gpu
{

//! Utility for attempting to estimate safe block/grid dimensions from working image dimensions
//! These are not necesserily optimal. Far from it.
template<typename T>
inline void InitDimFromOutputImage(dim3& blockDim, dim3& gridDim, const Image<T>& image, int blockx = 16, int blocky = 16)
{
    blockDim = dim3(boost::math::gcd<unsigned>(image.w,blockx), boost::math::gcd<unsigned>(image.h,blocky), 1);
    gridDim =  dim3( image.w / blockDim.x, image.h / blockDim.y, 1);
}

//////////////////////////////////////////////////////
// Create Matlab Lookup table
//////////////////////////////////////////////////////

__global__ void KernCreateMatlabLookupTable(
    Image<float2> lookup, float fu, float fv, float u0, float v0, float k1, float k2
) {
    const uint u = blockIdx.x*blockDim.x + threadIdx.x;
    const uint v = blockIdx.y*blockDim.y + threadIdx.y;

    const float pnu = (u-u0) / fu;
    const float pnv = (v-v0) / fv;
    const float r = sqrt(pnu*pnu + pnv*pnv);
    const float rr = r*r;
    const float rf = 1 + k1*rr + k2*rr*rr; // + k3*rr*rr*rr;

    lookup(u,v) = make_float2(
        (pnu*rf /*+ 2*p1*pn.x*pn.y + p2*(rr + 2*pn.x*pn.x)*/) * fu + u0,
        (pnv*rf /*+ p1*(rr + 2*pn.y*pn.y) + 2*p2*pn.x*pn.y*/) * fv + v0
    );
}

void CreateMatlabLookupTable(
    Image<float2> lookup, float fu, float fv, float u0, float v0, float k1, float k2
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, lookup);
    KernCreateMatlabLookupTable<<<gridDim,blockDim>>>(lookup,fu,fv,u0,v0,k1,k2);
}

//////////////////////////////////////////////////////
// Create Matlab Lookup table applying homography
//////////////////////////////////////////////////////

__global__ void KernCreateMatlabLookupTable(
    Image<float2> lookup, float fu, float fv, float u0, float v0, float k1, float k2, Array<float,9> H_on
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    // Apply homography H_on, moving New image to Original
    const float hdiv = H_on[6] * x + H_on[7] * y + H_on[8];
    const float u = (H_on[0] * x + H_on[1] * y + H_on[2]) / hdiv;
    const float v = (H_on[3] * x + H_on[4] * y + H_on[5]) / hdiv;

    // Apply distortion to achieve true image coordinates
    const float pnu = (u-u0) / fu;
    const float pnv = (v-v0) / fv;
    const float r = sqrt(pnu*pnu + pnv*pnv);
    const float rr = r*r;
    const float rf = 1 + k1*rr + k2*rr*rr; // + k3*rr*rr*rr;

    float2 pos = make_float2(
        (pnu*rf /*+ 2*p1*pn.x*pn.y + p2*(rr + 2*pn.x*pn.x)*/) * fu + u0,
        (pnv*rf /*+ p1*(rr + 2*pn.y*pn.y) + 2*p2*pn.x*pn.y*/) * fv + v0
    );

    // Clamp to image bounds
    pos.x = max(pos.x, 1.0f);
    pos.y = max(pos.y, 1.0f);
    pos.x = min(pos.x, lookup.w-2.0f);
    pos.y = min(pos.y, lookup.h-2.0f);

    lookup(x,y) = pos;
}

void CreateMatlabLookupTable(
        Image<float2> lookup, float fu, float fv, float u0, float v0, float k1, float k2, Array<float,9> H_no
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, lookup);
    KernCreateMatlabLookupTable<<<gridDim,blockDim>>>(lookup,fu,fv,u0,v0,k1,k2,H_no);
}

//////////////////////////////////////////////////////
// Warp image using lookup table
//////////////////////////////////////////////////////

__global__ void KernWarp(
    Image<unsigned char> out, const Image<unsigned char> in, const Image<float2> lookup
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const float2 lu = lookup(x,y);
    out(x,y) = bicubic<float,unsigned char>((unsigned char*)in.ptr, in.stride, lu.x, lu.y);
}

void Warp(
    Image<unsigned char> out, const Image<unsigned char> in, const Image<float2> lookup
) {
    assert(out.w <= lookup.w && out.h <= lookup.h);
    assert(out.w <= in.w && out.h <= in.w);

    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, out);
    KernWarp<<<gridDim,blockDim>>>(out, in, lookup);

}

//////////////////////////////////////////////////////
// Scanline rectified dense stereo
//////////////////////////////////////////////////////

template<typename T, int size>
__device__ inline
float Score(
    Image<T> img1, int x1, int y1,
    Image<T> img2, int x2, int y2
) {
    float sum_abs_diff = 0;

    for(int r=-size; r <=size; ++r ) {
        for(int c=-size; c <=size; ++c ) {
            float i1 = img1.GetWithClampedRange(x1+c,y1+r);
            float i2 = img2.GetWithClampedRange(x2+c,y2+r);
            sum_abs_diff += abs(i1 - i2);
        }
    }

    return sum_abs_diff / (size*size);
}

template<typename T, int MAX_DISP>
__global__ void KernDenseStereo(
    Image<float> dDisp, Image<T> dCamLeft, Image<T> dCamRight, int disp
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

//    dDisp(x,y) = abs(dCamLeft(x,y).x - dCamRight(x,y).x) / 255.0f;
//    dDisp(x,y) = Score<unsigned char,2>(dCamLeft, x,y, dCamRight, x-disp, y) / 255.0f;

    // Search for best matching pixel
    int bestDisp = MAX_DISP;
    float bestScore = 1E10;

    for(int c = 0; c < MAX_DISP; ++c ) {
        const float score = Score<unsigned char,2>(dCamLeft, x,y, dCamRight, x-c, y);
        if(score < bestScore) {
            bestScore = score;
            bestDisp = c;
        }
    }

    dDisp(x,y) = (float)bestDisp / (float)MAX_DISP;
}

void DenseStereo(
    Image<float> dDisp, Image<unsigned char> dCamLeft, Image<unsigned char> dCamRight, int disp
) {
//    const int MAX_DISP = 128;
//    dim3 blockDim, gridDim;
//    InitDimFromOutputImage(blockDim,gridDim, dDisp);
    dim3 blockDim(16,16);
    dim3 gridDim(dDisp.w / blockDim.x,dDisp.h / blockDim.y );
    KernDenseStereo<unsigned char,50><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight,disp);
}

//////////////////////////////////////////////////////
// Quick and dirty Bilateral filer
//////////////////////////////////////////////////////

__global__ void KernBilateralFilter(
    Image<float> dOut, Image<float> dIn, float gs, float gr, int size
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const float p = dIn(x,y);
    float sum = 0;
    float sumw = 0;

    for(int r = -size; r <= size; ++r ) {
        for(int c = -size; c <= size; ++c ) {
            const float q = dIn.GetWithClampedRange(x+c, y+r);
            const float sd2 = r*r + c*c;
            const float id = abs(p - q);
            const float sw = __expf(-(sd2) / (2 * gs * gs));
            const float iw = __expf(-(id * id) / (2 * gr * gr));
            const float w = sw*iw;
            sumw += w;
            sum += w * q;
//            sumw += 1;
//            sum += q;
        }
    }

    dOut(x,y) = sum / sumw;
}

void BilateralFilter(
    Image<float> dOut, Image<float> dIn, float gs, float gr, uint size
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut);
    KernBilateralFilter<<<gridDim,blockDim>>>(dOut, dIn, gs, gr, size);
}

//////////////////////////////////////////////////////
// Anaglyph: Join left / right images into anagly stereo
//////////////////////////////////////////////////////

__global__ void KernMakeAnaglyth(Image<uchar4> anaglyth, const Image<unsigned char> left, const Image<unsigned char> right)
{
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    anaglyth(x,y) = make_uchar4(left(x,y), 0, right(x,y),255);
}

void MakeAnaglyth(Image<uchar4> anaglyth, const Image<unsigned char> left, const Image<unsigned char> right)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, anaglyth);
    KernMakeAnaglyth<<<gridDim,blockDim>>>(anaglyth, left, right);
}

}

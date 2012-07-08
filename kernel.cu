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

namespace Gpu
{

//////////////////////////////////////////////////////
// Image Conversion
//////////////////////////////////////////////////////

template<typename To, typename Ti>
__host__ __device__
To ConvertPixel(Ti p)
{
    return p;
}

template<>
__host__ __device__
uchar4 ConvertPixel(unsigned char p)
{
    return make_uchar4(p,p,p,255);
}

template<typename To, typename Ti>
__global__
void KernConvertImage(Image<To> dOut, const Image<Ti> dIn)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    dOut(x,y) = ConvertPixel<To,Ti>(dIn(x,y));
}

template<typename To, typename Ti>
void ConvertImage(Image<To> dOut, const Image<Ti> dIn)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dOut);
    KernConvertImage<<<gridDim,blockDim>>>(dOut,dIn);
}

// Explicit instantiation
template void ConvertImage<uchar4,unsigned char>(Image<uchar4>, const Image<unsigned char>);
template void ConvertImage<float,unsigned char>(Image<float>, const Image<unsigned char>);
template void ConvertImage<float,char>(Image<float>, const Image<char>);

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
// Patch Scores
//////////////////////////////////////////////////////

template<typename T, int size>
__device__ inline
float Sum(
    Image<T> img, int x, int y
) {
    float sum = 0;
    for(int r=-size; r <=size; ++r ) {
        for(int c=-size; c <=size; ++c ) {
            sum += img.GetWithClampedRange(x+c,y+r);
        }
    }
    return sum;
}

// Mean Absolute Difference
template<typename To, typename T, int rad>
__device__ inline
float MADScore(
    Image<T> img1, int x1, int y1,
    Image<T> img2, int x2, int y2
) {
    const int w = 2*rad+1;
    To sum_abs_diff = 0;

    for(int r=-rad; r <=rad; ++r ) {
        for(int c=-rad; c <=rad; ++c ) {
            T i1 = img1.GetWithClampedRange(x1+c,y1+r);
            T i2 = img2.GetWithClampedRange(x2+c,y2+r);
            sum_abs_diff += abs(i1 - i2);
        }
    }

    return sum_abs_diff / (w*w);
}

// Sum Square Normalised Difference
template<typename To, typename T, int rad>
__device__ inline
To SSNDScore(
    Image<T> img1, int x1, int y1,
    Image<T> img2, int x2, int y2
) {
//    // Straightforward approach
//    const int w = 2*rad+1;
//    const float m1 = Sum<T,rad>(img1,x1,y1) / (w*w);
//    const float m2 = Sum<T,rad>(img2,x2,y2) / (w*w);

//    float sum_abs_diff = 0;
//    for(int r=-rad; r <=rad; ++r ) {
//        for(int c=-rad; c <=rad; ++c ) {
//            float i1 = img1.GetWithClampedRange(x1+c,y1+r) - m1;
//            float i2 = img2.GetWithClampedRange(x2+c,y2+r) - m2;
//            sum_abs_diff += abs(i1 - i2);
//        }
//    }
//    return sum_abs_diff;

    To sxi = 0;
    To sxi2 = 0;
    To syi = 0;
    To syi2 = 0;
    To sxiyi = 0;

    const int w = 2*rad+1;
    const int n = w*w;

    for(int r=-rad; r <=rad; ++r ) {
        for(int c=-rad; c <=rad; ++c ) {
            To xi = img1.GetWithClampedRange(x1+c,y1+r);
            To yi = img2.GetWithClampedRange(x2+c,y2+r);
            sxi += xi;
            syi += yi;
            sxi2 += xi*xi;
            syi2 += yi*yi;
            sxiyi += xi*yi;
        }
    }

    const To mx = (float)sxi / (float)n;
    const To my = (float)syi / (float)n;

    const To score = 0
            + sxi2 - 2*mx*sxi + n*mx*mx
            + 2*(-sxiyi + my*sxi + mx*syi - n*mx*my)
            + syi2 - 2*my*syi + n*my*my;
    return score;
}

//////////////////////////////////////////////////////
// Scanline rectified dense stereo
//////////////////////////////////////////////////////

template<typename TD, typename TI, unsigned int rad>
__global__ void KernDenseStereo(
    Image<TD> dDisp, Image<TI> dCamLeft, Image<TI> dCamRight, int maxDisp
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    // Search for best matching pixel
    int bestDisp = 0;
    float bestScore = 1E10;
    float sndBestScore = 1E11;

    for(int c = 0; c <= maxDisp; ++c ) {
        const float score = SSNDScore<float,TI,rad>(dCamLeft, x,y, dCamRight, x-c, y);
        if(score < bestScore) {
            sndBestScore = bestScore;
            bestScore = score;
            bestDisp = c;
        }else if( score < sndBestScore) {
            sndBestScore = score;
        }
    }

    const bool valid = true; //(bestScore * 1.2) < sndBestScore;

    dDisp(x,y) = valid ? bestDisp : -1;
}

void DenseStereo(
    Image<char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, int maxDisp
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dDisp);
    KernDenseStereo<char, unsigned char, 3><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight,maxDisp);
}

//////////////////////////////////////////////////////
// Scanline rectified dense stereo sub-pixel refinement
//////////////////////////////////////////////////////

template<typename TDo, typename TDi, typename TI, unsigned int rad>
__global__ void KernDenseStereoSubpixelRefine(
    Image<TDo> dDispOut, const Image<TDi> dDisp, const Image<TI> dCamLeft, const Image<TI> dCamRight
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const int bestDisp = dDisp(x,y);

    if(bestDisp == -1) {
        dDispOut(x,y) = -1;
        return;
    }

    // Fit parabola to neighbours
    const float d1 = bestDisp+1;
    const float d2 = bestDisp;
    const float d3 = bestDisp-1;
    const float s1 = SSNDScore<float,unsigned char,rad>(dCamLeft, x,y, dCamRight, x-d1,y);
    const float s2 = SSNDScore<float,unsigned char,rad>(dCamLeft, x,y, dCamRight, x-d2,y);
    const float s3 = SSNDScore<float,unsigned char,rad>(dCamLeft, x,y, dCamRight, x-d3,y);

    // Cooefficients of parabola through (d1,s1),(d2,s2),(d3,s3)
    const float denom = (d1 - d2)*(d1 - d3)*(d2 - d3);
    const float A = (d3 * (s2 - s1) + d2 * (s1 - s3) + d1 * (s3 - s2)) / denom;
    const float B = (d3*d3 * (s1 - s2) + d2*d2 * (s3 - s1) + d1*d1 * (s2 - s3)) / denom;
//    const float C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

    // Minima of parabola
    const float newDisp = -B / (2*A);

    // Check that minima is sensible. Otherwise assume bad data.
    if( d3 < newDisp && newDisp < d1 ) {
        dDispOut(x,y) = newDisp;
    }else{
//        dDisp(x,y) = bestDisp / maxDisp;
        dDispOut(x,y) = -1;
    }
}

void DenseStereoSubpixelRefine(
    Image<float> dDispOut, const Image<char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dDisp);
    KernDenseStereoSubpixelRefine<float,char,unsigned char,3><<<gridDim,blockDim>>>(dDispOut, dDisp, dCamLeft, dCamRight);
}

//////////////////////////////////////////////////////
// Upgrade disparity image to vertex array
//////////////////////////////////////////////////////

__global__ void KernDisparityImageToVbo(
    Image<float4> dVbo, const Image<float> dDisp, double baseline, double fu, double fv, double u0, double v0
) {
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    const float disp = dDisp(u,v);
    const float z = disp >= 0 ? fu * baseline / -disp : -1E10;

    // (x,y,1) = kinv * (u,v,1)'
    const float x = z * (u-u0) / fu;
    const float y = z * (v-v0) / fv;

    dVbo(u,v) = make_float4(x,y,z,1);
}

void DisparityImageToVbo(Image<float4> dVbo, const Image<float> dDisp, double baseline, double fu, double fv, double u0, double v0)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dVbo);
    KernDisparityImageToVbo<<<gridDim,blockDim>>>(dVbo, dDisp, baseline, fu, fv, u0, v0);
}

//////////////////////////////////////////////////////
// Make Index Buffer for rendering
//////////////////////////////////////////////////////

__global__ void KernGenerateTriangleStripIndexBuffer(Image<uint2> dIbo)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    const unsigned int pixIndex = y*dIbo.w + x;
    dIbo(x,y) = make_uint2(pixIndex, pixIndex + dIbo.w);
}

void GenerateTriangleStripIndexBuffer( Image<uint2> dIbo)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dIbo);
    KernGenerateTriangleStripIndexBuffer<<<gridDim,blockDim>>>(dIbo);
}

//////////////////////////////////////////////////////
// Experimental and dirty Bilateral filer
//////////////////////////////////////////////////////

template<typename To, typename Ti>
__global__ void KernRobustBilateralFilter(
    Image<To> dOut, Image<Ti> dIn, float gs, float gr, float go, int size
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
    Image<float> dOut, Image<unsigned char> dIn, float gs, float gr, float go, uint size
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
    Image<To> dOut, Image<Ti> dIn, float gs, float gr, int size
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
    Image<float> dOut, Image<float> dIn, float gs, float gr, uint size
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut);
    KernBilateralFilter<float,float><<<gridDim,blockDim>>>(dOut, dIn, gs, gr, size);
}

void BilateralFilter(
    Image<float> dOut, Image<unsigned char> dIn, float gs, float gr, uint size
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut);
    KernBilateralFilter<float,unsigned char><<<gridDim,blockDim>>>(dOut, dIn, gs, gr, size);
}

//////////////////////////////////////////////////////
// Median Filter
//////////////////////////////////////////////////////

// Exchange trick: Morgan McGuire, ShaderX 2008
#define s2(a,b)            { float tmp = a; a = min(a,b); b = max(tmp,b); }
#define mn3(a,b,c)         s2(a,b); s2(a,c);
#define mx3(a,b,c)         s2(b,c); s2(a,c);

#define mnmx3(a,b,c)       mx3(a,b,c); s2(a,b);                               // 3 exchanges
#define mnmx4(a,b,c,d)     s2(a,b); s2(c,d); s2(a,c); s2(b,d);                // 4 exchanges
#define mnmx5(a,b,c,d,e)   s2(a,b); s2(c,d); mn3(a,c,e); mx3(b,d,e);          // 6 exchanges
#define mnmx6(a,b,c,d,e,f) s2(a,d); s2(b,e); s2(c,f); mn3(a,b,c); mx3(d,e,f); // 7 exchanges

#define SMEM(x,y)  smem[(x)+1][(y)+1]
#define IN(x,y)    d_in[(y)*nx + (x)]

// http://blog.accelereyes.com/blog/2010/03/04/median-filtering-cuda-tips-and-tricks/
// Which in turn is based on http://graphics.cs.williams.edu/papers/MedianShaderX6/
template<int BLOCK_X, int BLOCK_Y>
__global__ void KernMedianFilter3x3(int nx, int ny, float *d_out, float *d_in)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    // guards: is at boundary?
    bool is_x_top = (tx == 0), is_x_bot = (tx == BLOCK_X-1);
    bool is_y_top = (ty == 0), is_y_bot = (ty == BLOCK_Y-1);

    __shared__ float smem[BLOCK_X+2][BLOCK_Y+2];
    // clear out shared memory (zero padding)
    if (is_x_top)           SMEM(tx-1, ty  ) = 0;
    else if (is_x_bot)      SMEM(tx+1, ty  ) = 0;
    if (is_y_top) {         SMEM(tx  , ty-1) = 0;
        if (is_x_top)       SMEM(tx-1, ty-1) = 0;
        else if (is_x_bot)  SMEM(tx+1, ty-1) = 0;
    } else if (is_y_bot) {  SMEM(tx  , ty+1) = 0;
        if (is_x_top)       SMEM(tx-1, ty+1) = 0;
        else if (is_x_bot)  SMEM(tx+1, ty+1) = 0;
    }

    // guards: is at boundary and still more image?
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    is_x_top &= (x > 0); is_x_bot &= (x < nx - 1);
    is_y_top &= (y > 0); is_y_bot &= (y < ny - 1);

    // each thread pulls from image
                            SMEM(tx  , ty  ) = IN(x  , y  ); // self
    if (is_x_top)           SMEM(tx-1, ty  ) = IN(x-1, y  );
    else if (is_x_bot)      SMEM(tx+1, ty  ) = IN(x+1, y  );
    if (is_y_top) {         SMEM(tx  , ty-1) = IN(x  , y-1);
        if (is_x_top)       SMEM(tx-1, ty-1) = IN(x-1, y-1);
        else if (is_x_bot)  SMEM(tx+1, ty-1) = IN(x+1, y-1);
    } else if (is_y_bot) {  SMEM(tx  , ty+1) = IN(x  , y+1);
        if (is_x_top)       SMEM(tx-1, ty+1) = IN(x-1, y+1);
        else if (is_x_bot)  SMEM(tx+1, ty+1) = IN(x+1, y+1);
    }
    __syncthreads();

    // pull top six from shared memory
    float v[6] = { SMEM(tx-1, ty-1), SMEM(tx  , ty-1), SMEM(tx+1, ty-1),
                   SMEM(tx-1, ty  ), SMEM(tx  , ty  ), SMEM(tx+1, ty  ) };

    // with each pass, remove min and max values and add new value
    mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
    v[5] = SMEM(tx-1, ty+1); // add new contestant
    mnmx5(v[1], v[2], v[3], v[4], v[5]);
    v[5] = SMEM(tx  , ty+1);
    mnmx4(v[2], v[3], v[4], v[5]);
    v[5] = SMEM(tx+1, ty+1);
    mnmx3(v[3], v[4], v[5]);

    // pick the middle one
    d_out[y*nx + x] = v[4];
}

void MedianFilter3x3(
    Image<float> dOut, Image<float> dIn
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut, 16, 16);
    KernMedianFilter3x3<16,16><<<gridDim,blockDim>>>((int)dOut.w, (int)dOut.h, dOut.ptr, dIn.ptr);
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

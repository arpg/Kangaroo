#include "cu_painting.h"

#include "launch_utils.h"

namespace roo {

//////////////////////////////////////////////////////
// Paint circle at x,y with radius r
//////////////////////////////////////////////////////

template<typename T>
__global__ void KernPaintCircle(Image<T> img, T val, float cx, float cy, float cr )
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    const float dx = (x-cx);
    const float dy = (y-cy);
    const float r_sq = dx*dx + dy*dy;

    if( x < img.w && y < img.h && r_sq < cr*cr ) {
        img(x,y) = val;
    }
}

template<typename T>
void PaintCircle(Image<T> img, T val, float x, float y, float r )
{
    const float safe_r = r+1;
    const int sx = fminf(img.w-1, max(0.0f, floor(x-safe_r) ));
    const int sy = fminf(img.h-1, max(0.0f, floor(y-safe_r) ));
    const int w = fmaxf(fminf(img.w-sx, ceil(x+safe_r) - sx), 0);
    const int h = fmaxf(fminf(img.h-sy, ceil(y+safe_r) - sy), 0);

    dim3 blockDim, gridDim;
    blockDim = dim3(32, 32);
    gridDim =  dim3( ceil(w / (double)blockDim.x), ceil(h / (double)blockDim.y) );

    KernPaintCircle<T><<<gridDim,blockDim>>>(img.SubImage(sx,sy,w,h),val,x-sx,y-sy,r);
}

//////////////////////////////////////////////////////
// Template instantiations
//////////////////////////////////////////////////////

template void PaintCircle(Image<float> img, float val, float x, float y, float r );
template void PaintCircle(Image<unsigned char> img, unsigned char val, float x, float y, float r );

}

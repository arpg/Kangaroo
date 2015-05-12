#include "cu_integral_image.h"

#include "launch_utils.h"
#include "CUDA_SDK/sharedmem.h"

namespace roo
{

//////////////////////////////////////////////////////
// Image Transpose
// Efficient Integral Image Computation on the GPU
// Berkin Bilgic, Berthold K.P. Horn, Ichiro Masaki
//////////////////////////////////////////////////////

template<typename Tout, typename Tin, int BLOCK_DIM>
__global__ void KernTranspose(Image<Tout> out, Image<Tin> in)
{
    __shared__ Tin temp[BLOCK_DIM][BLOCK_DIM+1];
    int xIndex = blockIdx.x*BLOCK_DIM + threadIdx.x;
    int yIndex = blockIdx.y*BLOCK_DIM + threadIdx.y;

    if((xIndex < in.w) && (yIndex < in.h)) {
        temp[threadIdx.y][threadIdx.x] = in(xIndex,yIndex);
    }

    __syncthreads();

    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

    if((xIndex < in.h) && (yIndex < in.w)) {
        out(xIndex,yIndex) = temp[threadIdx.x][threadIdx.y];
    }
}

template<typename Tout, typename Tin>
void Transpose(Image<Tout> out, Image<Tin> in)
{
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim,in, 16,16);
    KernTranspose<Tout,Tin,16><<<gridDim,blockDim>>>(out,in);
}

// Instantiate useful versions
template KANGAROO_EXPORT void Transpose(Image<unsigned char>,Image<unsigned char>);
template KANGAROO_EXPORT void Transpose(Image<int>,Image<int>);
template KANGAROO_EXPORT void Transpose(Image<float>,Image<float>);

//////////////////////////////////////////////////////
// PrefixSum
// Efficient Integral Image Computation on the GPU
// Berkin Bilgic, Berthold K.P. Horn, Ichiro Masaki
//////////////////////////////////////////////////////

template<typename Tout, typename Tin>
inline __device__
void PrefixSum(Tout* output, Tin* input, int w, int nextpow2)
{
    SharedMemory<Tout> shared;
    Tout* temp = shared.getPointer();

    const int tdx = threadIdx.x;
    int offset = 1;
    const int tdx2 = 2*tdx;
    const int tdx2p = tdx2 + 1;

    temp[tdx2] =  tdx2 < w ? input[tdx2] : 0;
    temp[tdx2p] = tdx2p < w ? input[tdx2p] : 0;

    for(int d = nextpow2>>1; d > 0; d >>= 1) {
        __syncthreads();
        if(tdx < d)
        {
            int ai = offset*(tdx2p)-1;
            int bi = offset*(tdx2+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if(tdx == 0) temp[nextpow2 - 1] = 0;

    for(int d = 1; d < nextpow2; d *= 2) {
        offset >>= 1;

        __syncthreads();

        if(tdx < d )
        {
            int ai = offset*(tdx2p)-1;
            int bi = offset*(tdx2+2)-1;
            Tout t  = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    if(tdx2 < w)  output[tdx2] = temp[tdx2];
    if(tdx2p < w) output[tdx2p] = temp[tdx2p];
}

template<typename Tout, typename Tin>
__global__ void KernPrefixSumRows(Image<Tout> out, Image<Tin> in)
{
    const int row = blockIdx.y;
    PrefixSum<Tout,Tin>(out.RowPtr(row), in.RowPtr(row), in.w, 2*blockDim.x );
}

template<typename Tout, typename Tin>
void PrefixSumRows(Image<Tout> out, Image<Tin> in)
{
    dim3 blockDim = dim3( 1, 1);
    while(blockDim.x < ceil(in.w/2.0f)) blockDim.x <<= 1;
    const dim3 gridDim =  dim3( 1, in.h );
    KernPrefixSumRows<<<gridDim,blockDim,2*sizeof(Tout)*blockDim.x>>>(out,in);
}

// Instantiate useful versions
template KANGAROO_EXPORT void PrefixSumRows(Image<int>, Image<unsigned char>);
template KANGAROO_EXPORT void PrefixSumRows(Image<int>, Image<int>);
template KANGAROO_EXPORT void PrefixSumRows(Image<float>, Image<float>);

//////////////////////////////////////////////////////
// Large Radius Box Filter using Integral Image
//////////////////////////////////////////////////////

template<typename Tout, typename Tin>
__global__ void KernBoxFilterIntegralImage(Image<Tout> out, Image<Tin> IntegralImageT, int rad)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(out.InBounds(x,y)) {
        const int minx = max(0,x-rad);
        const int maxx = min((int)out.w-1,x+rad);
        const int miny = max(0,y-rad);
        const int maxy = min((int)out.h-1,y+rad);

        const int winw = maxx - minx;
        const int winh = maxy - miny;
        const int area = winw * winh;

        const Tin A = IntegralImageT(miny,minx);
        const Tin B = IntegralImageT(miny,maxx);
        const Tin C = IntegralImageT(maxy,maxx);
        const Tin D = IntegralImageT(maxy,minx);

        const Tin sum = C+A-B-D;
        const Tout mean = (float)sum / area;
        out(x,y) = mean;
    }
}

template<typename Tout, typename Tin>
void BoxFilterIntegralImage(Image<Tout> out, Image<Tin> IntegralImageT, int rad)
{
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, out);
    KernBoxFilterIntegralImage<Tout,Tin><<<gridDim,blockDim>>>(out,IntegralImageT,rad);
}

// Instantiate useful versions
template KANGAROO_EXPORT void BoxFilterIntegralImage(Image<float>, Image<int>, int);
template KANGAROO_EXPORT void BoxFilterIntegralImage(Image<float>, Image<float>, int);


}

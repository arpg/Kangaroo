#include "all.h"
#include "launch_utils.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Median Filter
//////////////////////////////////////////////////////

// Exchange trick: Morgan McGuire, ShaderX 2008

template<typename T> inline __host__ __device__
void s2(T& a, T& b) { T tmp = a; a = min(a,b); b = max(tmp,b); }

template<typename T> inline __host__ __device__
void mn3(T& a, T& b, T& c) { s2(a,b); s2(a,c); }

template<typename T> inline __host__ __device__
void mx3(T& a, T& b, T& c) { s2(b,c); s2(a,c); }

template<typename T> inline __host__ __device__
void mnmx3(T& a, T& b, T& c) { mx3(a,b,c); s2(a,b); }

template<typename T> inline __host__ __device__
void mnmx4(T& a,T& b,T& c,T& d) { s2(a,b); s2(c,d); s2(a,c); s2(b,d); }

template<typename T> inline __host__ __device__
void mnmx5(T& a,T& b,T& c,T& d,T& e) { s2(a,b); s2(c,d); mn3(a,c,e); mx3(b,d,e); }

template<typename T> inline __host__ __device__
void mnmx6(T& a,T& b,T& c,T& d,T& e,T& f) { s2(a,d); s2(b,e); s2(c,f); mn3(a,b,c); mx3(d,e,f); }

#define SMEM(x,y)  smem[(x)+1][(y)+1]


// http://blog.accelereyes.com/blog/2010/03/04/median-filtering-cuda-tips-and-tricks/
// Which in turn is based on http://graphics.cs.williams.edu/papers/MedianShaderX6/
template<typename To, typename Ti, int BLOCK_X, int BLOCK_Y>
__global__ void KernMedianFilter3x3(Image<To> dOut, Image<Ti> dIn )
{
    const int tx = threadIdx.x, ty = threadIdx.y;

    // guards: is at boundary?
    bool is_x_top = (tx == 0), is_x_bot = (tx == BLOCK_X-1);
    bool is_y_top = (ty == 0), is_y_bot = (ty == BLOCK_Y-1);

    __shared__ Ti smem[BLOCK_X+2][BLOCK_Y+2];
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
    is_x_top &= (x > 0); is_x_bot &= (x < dOut.w - 1);
    is_y_top &= (y > 0); is_y_bot &= (y < dOut.h - 1);

    // each thread pulls from image
                            SMEM(tx  , ty  ) = dIn(x  , y  ); // self
    if (is_x_top)           SMEM(tx-1, ty  ) = dIn(x-1, y  );
    else if (is_x_bot)      SMEM(tx+1, ty  ) = dIn(x+1, y  );
    if (is_y_top) {         SMEM(tx  , ty-1) = dIn(x  , y-1);
        if (is_x_top)       SMEM(tx-1, ty-1) = dIn(x-1, y-1);
        else if (is_x_bot)  SMEM(tx+1, ty-1) = dIn(x+1, y-1);
    } else if (is_y_bot) {  SMEM(tx  , ty+1) = dIn(x  , y+1);
        if (is_x_top)       SMEM(tx-1, ty+1) = dIn(x-1, y+1);
        else if (is_x_bot)  SMEM(tx+1, ty+1) = dIn(x+1, y+1);
    }
    __syncthreads();

    // pull top six from shared memory
    Ti v[6] = { SMEM(tx-1, ty-1), SMEM(tx  , ty-1), SMEM(tx+1, ty-1),
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
    dOut(x,y) = v[4];
}

void MedianFilter3x3(
    Image<float> dOut, Image<float> dIn
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut, 16, 16);
    KernMedianFilter3x3<float,float,16,16><<<gridDim,blockDim>>>(dOut, dIn);
}

#define t2(a, b)                            s2(v[a], v[b]);
#define t24(a, b, c, d, e, f, g, h)			t2(a, b); t2(c, d); t2(e, f); t2(g, h);
#define t25(a, b, c, d, e, f, g, h, i, j)	t24(a, b, c, d, e, f, g, h); t2(i, j);

// Based on fragment shader: http://graphics.cs.williams.edu/papers/MedianShaderX6/median5.pix
template<typename To, typename Ti, int BLOCK_X, int BLOCK_Y>
__global__ void KernMedianFilter5x5(Image<To> dOut, Image<Ti> dIn )
{
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    Ti v[25];

    // TODO: Distribute this accross block with apron
    for(int dX = -2; dX <= 2; ++dX) {
      for(int dY = -2; dY <= 2; ++dY) {
          v[(dX + 2) * 5 + (dY + 2)] = dIn.GetWithClampedRange(x+dX,y+dY);
      }
    }

    t25(0, 1,	3, 4,	2, 4,	2, 3,	6, 7);
    t25(5, 7,	5, 6,	9, 7,	1, 7,	1, 4);
    t25(12, 13,	11, 13,	11, 12,	15, 16,	14, 16);
    t25(14, 15,	18, 19,	17, 19,	17, 18,	21, 22);
    t25(20, 22,	20, 21,	23, 24,	2, 5,	3, 6);
    t25(0, 6,	0, 3,	4, 7,	1, 7,	1, 4);
    t25(11, 14,	8, 14,	8, 11,	12, 15,	9, 15);
    t25(9, 12,	13, 16,	10, 16,	10, 13,	20, 23);
    t25(17, 23,	17, 20,	21, 24,	18, 24,	18, 21);
    t25(19, 22,	8, 17,	9, 18,	0, 18,	0, 9);
    t25(10, 19,	1, 19,	1, 10,	11, 20,	2, 20);
    t25(2, 11,	12, 21,	3, 21,	3, 12,	13, 22);
    t25(4, 22,	4, 13,	14, 23,	5, 23,	5, 14);
    t25(15, 24,	6, 24,	6, 15,	7, 16,	7, 19);
    t25(3, 11,	5, 17,	11, 17,	9, 17,	4, 10);
    t25(6, 12,	7, 14,	4, 6,	4, 7,	12, 14);
    t25(10, 14,	6, 7,	10, 12,	6, 10,	6, 17);
    t25(12, 17,	7, 17,	7, 10,	12, 18,	7, 12);
    t24(10, 18,	12, 20,	10, 20,	10, 12);

    dOut(x,y) = (To)v[12];
}

void MedianFilter5x5(
    Image<float> dOut, Image<float> dIn
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut, 16, 16);
    KernMedianFilter5x5<float,float,16,16><<<gridDim,blockDim>>>(dOut, dIn);
}

}

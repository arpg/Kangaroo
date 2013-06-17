#pragma once

#include "Mat.h"
#include "Image.h"

namespace roo
{

///////////////////////////////////////////
// Sum Linear Systems in shared memory
// __shared__ SumLeastSquaresSystem<float,6,16,16> sumlss;
// LeastSquaresSystem<T,N>& lss = sumlss.ThisObs();
// ...
// sumlss.ReducePutBlock(dSum);
///////////////////////////////////////////

template<typename T, unsigned N>
struct HostSumLeastSquaresSystem
{
    Image<LeastSquaresSystem<T,N> > dSum;

    __host__ HostSumLeastSquaresSystem(Image<unsigned char>& dWorkspace, dim3 blockDim, dim3 gridDim)
    {
        dSum = dWorkspace.PackedImage<LeastSquaresSystem<T,N> >(gridDim.x, gridDim.y);
    }

    __host__ inline Image<LeastSquaresSystem<T,N> >& LeastSquareImage()
    {
        return dSum;
    }

    __host__ inline LeastSquaresSystem<T,N> FinalSystem()
    {
        LeastSquaresSystem<T,N> sum;
        sum.SetZero();
        return thrust::reduce(dSum.begin(), dSum.end(), sum, thrust::plus<LeastSquaresSystem<T,N> >() );
    }
};


template<typename T, unsigned N, unsigned MAX_BLOCK_X, unsigned MAX_BLOCK_Y>
struct SumLeastSquaresSystem
{
    LeastSquaresSystem<T,N> sReduce[MAX_BLOCK_X * MAX_BLOCK_Y];

    __host__ static inline Image<LeastSquaresSystem<T,N> > CreateSumImage(Image<unsigned char>& dWorkspace, dim3 blockDim, dim3 gridDim)
    {
        return dWorkspace.PackedImage<LeastSquaresSystem<T,N> >(gridDim.x, gridDim.y);
    }

    __host__ static inline LeastSquaresSystem<T,N> FinalSum(Image<LeastSquaresSystem<T,N> >& dSum)
    {
        LeastSquaresSystem<T,N> sum;
        sum.SetZero();
        return thrust::reduce(dSum.begin(), dSum.end(), sum, thrust::plus<LeastSquaresSystem<T,N> >() );
    }

    __device__ inline LeastSquaresSystem<T,N>& ThisObs()
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        return sReduce[tid];
    }

    __device__ inline LeastSquaresSystem<T,N>& ZeroThisObs()
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        sReduce[tid].SetZero();
        return sReduce[tid];
    }

    __device__ inline void ReducePutBlock(Image<LeastSquaresSystem<T,N> >& dSum)
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        const unsigned int bid = blockIdx.y*gridDim.x + blockIdx.x;
        __syncthreads();
        for(unsigned S=blockDim.y*blockDim.x/2;S>0; S>>=1)  {
            if( tid < S ) {
                sReduce[tid] += sReduce[tid+S];
            }
            __syncthreads();
        }
        if( tid == 0) {
            dSum[bid] = sReduce[0];
        }
    }
};

}

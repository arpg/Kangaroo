#pragma once

#include <boost/math/common_factor.hpp>

#include "Image.h"

namespace Gpu
{

//! Utility for attempting to estimate safe block/grid dimensions from working image dimensions
//! These are not necesserily optimal. Far from it.
template<typename T, typename Target, typename Management>
inline void InitDimFromOutputImage(dim3& blockDim, dim3& gridDim, const Image<T,Target,Management>& image, int blockx = 32, int blocky = 32)
{
    blockDim = dim3(boost::math::gcd<unsigned>(image.w,blockx), boost::math::gcd<unsigned>(image.h,blocky), 1);
    gridDim =  dim3( image.w / blockDim.x, image.h / blockDim.y, 1);
}

//! Utility for attempting to estimate safe block/grid dimensions from working image dimensions
//! These are not necesserily optimal. Far from it.
template<typename T, typename Target, typename Management>
inline void InitDimFromOutputImageOver(dim3& blockDim, dim3& gridDim, const Image<T,Target,Management>& image, int blockx = 32, int blocky = 32)
{
    blockDim = dim3(blockx, blocky);
    gridDim =  dim3( ceil(image.w / (double)blockDim.x), ceil(image.h / (double)blockDim.y) );
}

}

#include <boost/math/common_factor.hpp>

#include "Image.h"

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

}

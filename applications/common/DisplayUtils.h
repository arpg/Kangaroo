#pragma once

#include <pangolin/glcuda.h>
#include <kangaroo/Image.h>


template<typename T, typename Owner>
inline void CopyToTex(pangolin::GlTextureCudaArray& tex, const Gpu::Image<T,Gpu::TargetDevice,Owner>& dImg)
{
    pangolin::CudaScopedMappedArray arr_tex(tex);
    cudaError_t err = cudaMemcpy2DToArray(*arr_tex, 0, 0, dImg.ptr, dImg.pitch, sizeof(T)*min(tex.width,dImg.w), min(tex.height,dImg.h), cudaMemcpyDeviceToDevice );
    if( err != cudaSuccess ) {
        std::cerr << "cudaMemcpy2DToArray failed: " << err << std::endl;
    }
}

template<typename T, typename Owner>
inline void operator<<(pangolin::GlTextureCudaArray& tex, const Gpu::Image<T,Gpu::TargetDevice,Owner>& dImg)
{
    CopyToTex(tex,dImg);
}

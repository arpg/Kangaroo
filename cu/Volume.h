#pragma once

#include <cuda_runtime.h>

#define HAVE_THRUST
#ifdef HAVE_THRUST
#include <thrust/device_vector.h>
#endif // HAVE_THRUST

#define HAVE_NPP
#ifdef HAVE_NPP
#include <npp.h>
#endif // HAVE_NPP

#include "Image.h"
#include "Mat.h"
#include "sampling.h"

namespace Gpu
{

template<typename T, typename Target = TargetDevice, typename Management = DontManage>
struct Volume
{
    inline __device__ __host__
    ~Volume()
    {
        Management::template Cleanup<T,Target>(ptr);
    }

    //////////////////////////////////////////////////////
    // Constructors
    //////////////////////////////////////////////////////

    template<typename ManagementCopyFrom> inline __host__ __device__
    Volume( const Volume<T,Target,ManagementCopyFrom>& img )
        : pitch(img.pitch), ptr(img.ptr), w(img.w), h(img.h), img_pitch(img.img_pitch), d(img.d)
    {
        Management::AssignmentCheck();
    }

    inline __host__
    Volume()
        : pitch(0), ptr(0), w(0), h(0), img_pitch(0), d(0)
    {
    }

    inline __host__
    Volume(unsigned int w, unsigned int h, unsigned int d)
        :w(w), h(h), d(d)
    {
        Management::AllocateCheck();
        Target::template AllocatePitchedMem<T>(&ptr,&pitch,w,h*d);
        img_pitch = pitch*h;
    }

    inline __device__ __host__
    Volume(T* ptr, size_t w, size_t h, size_t d)
        : pitch(sizeof(T)*w), ptr(ptr), w(w), h(h), img_pitch(sizeof(T)*w*h)
    {
    }

    inline __device__ __host__
    Volume(T* ptr, size_t w, size_t h, size_t d, size_t pitch)
        : pitch(pitch), ptr(ptr), w(w), h(h), img_pitch(pitch*h), d(d)
    {
    }

    inline __device__ __host__
    Volume(T* ptr, size_t w, size_t h, size_t d, size_t pitch, size_t img_pitch)
        : pitch(pitch), ptr(ptr), w(w), h(h), img_pitch(img_pitch), d(d)
    {
    }

    //////////////////////////////////////////////////////
    // Image copy
    //////////////////////////////////////////////////////

    template<typename TargetFrom, typename ManagementFrom>
    inline __host__
    void CopyFrom(const Volume<T,TargetFrom,ManagementFrom>& img)
    {
        // If these volumes don't have the same height, or have an image pitch different from their height,
        // we need to do a copy for each depth layer.
        assert(h = img.h && img_pitch = pitch*h && img.img_pitch == img.pitch * img.h);
        cudaMemcpy2D(ptr,pitch,img.ptr,img.pitch, std::min(img.w,w)*sizeof(T), h*std::min(img.d,d), TargetCopyKind<Target,TargetFrom>() );
    }

    template <typename DT>
    inline __host__
    void MemcpyFromHost(DT* hptr, size_t hpitch )
    {
        cudaMemcpy2D( (void*)ptr, pitch, hptr, hpitch, w*sizeof(T), h*d, cudaMemcpyHostToDevice );
    }

    template <typename DT>
    inline __host__
    void MemcpyFromHost(DT* ptr )
    {
        MemcpyFromHost(ptr, w*sizeof(T) );
    }

    //////////////////////////////////////////////////////
    // Direct Pixel Access
    //////////////////////////////////////////////////////

    inline  __device__ __host__
    const T* ImagePtr(size_t z) const
    {
        return (T*)((unsigned char*)(ptr) + z*img_pitch);
    }

    inline  __device__ __host__
    T* ImagePtr(size_t z)
    {
        return (T*)((unsigned char*)(ptr) + z*img_pitch);
    }

    inline  __device__ __host__
    T* RowPtr(size_t y, size_t z)
    {
        return (T*)((unsigned char*)(ptr) + z*img_pitch + y*pitch);
    }

    inline  __device__ __host__
    const T* RowPtr(size_t y, size_t z) const
    {
        return (T*)((unsigned char*)(ptr) + z*img_pitch + y*pitch);
    }

    inline  __device__ __host__
    T& operator()(size_t x, size_t y, size_t z)
    {
        return RowPtr(y,z)[x];
    }

    inline  __device__ __host__
    const T& operator()(size_t x, size_t y, size_t z) const
    {
        return RowPtr(y,z)[x];
    }

    inline  __device__ __host__
    T& operator[](size_t ix)
    {
        return ptr[ix];
    }

    inline  __device__ __host__
    const T& operator[](size_t ix) const
    {
        return ptr[ix];
    }

    inline  __device__ __host__
    const T& Get(int x, int y, int z) const
    {
        return RowPtr(y,z)[x];
    }

    //////////////////////////////////////////////////////
    // Obtain slices / subimages
    //////////////////////////////////////////////////////

    inline __device__ __host__
    const Image<T,Target,DontManage> ImageXY(size_t z) const
    {
        assert( z < d );
        return Image<T,Target,DontManage>( ImagePtr(z), w, h, pitch);
    }

    inline __device__ __host__
    const Image<T,Target,DontManage> ImageXZ(size_t y) const
    {
        assert( y < h );
        return Image<T,Target,DontManage>( RowPtr(y,0), w, d, img_pitch);
    }



    //////////////////////////////////////////////////////
    // Member variables
    //////////////////////////////////////////////////////

    size_t pitch;
    T* ptr;
    size_t w;
    size_t h;

    size_t img_pitch;
    size_t d;
};

}

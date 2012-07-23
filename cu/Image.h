#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

#include <iostream>
#include <assert.h>
#include <boost/static_assert.hpp>

#include <cuda_runtime.h>

#define HAVE_THRUST
#ifdef HAVE_THRUST
#include <thrust/device_vector.h>
#endif // HAVE_THRUST

#define HAVE_NPP
#ifdef HAVE_NPP
#include <npp.h>
#endif // HAVE_NPP

namespace Gpu
{

struct TargetHost
{
    template<typename T> inline static
    void AllocatePitchedMem(T** hostPtr, size_t *pitch, size_t w, size_t h){
        *pitch = w*sizeof(T);
        *hostPtr = (T*)malloc(*pitch * h);
    }

    template<typename T> inline static
    void DeallocatePitchedMem(T* hostPtr){
        free(hostPtr);
    }
};

struct TargetDevice
{
    template<typename T> inline static
    void AllocatePitchedMem(T** devPtr, size_t *pitch, size_t w, size_t h){
        cudaMallocPitch(devPtr,pitch,w*sizeof(T),h);
    }

    template<typename T> inline static
    void DeallocatePitchedMem(T* devPtr){
        cudaFree(devPtr);
    }
};

template<typename TargetTo, typename TargetFrom>
cudaMemcpyKind TargetCopyKind();

template<> inline cudaMemcpyKind TargetCopyKind<TargetHost,TargetHost>() { return cudaMemcpyHostToHost;}
template<> inline cudaMemcpyKind TargetCopyKind<TargetDevice,TargetHost>() { return cudaMemcpyHostToDevice;}
template<> inline cudaMemcpyKind TargetCopyKind<TargetHost,TargetDevice>() { return cudaMemcpyDeviceToHost;}
template<> inline cudaMemcpyKind TargetCopyKind<TargetDevice,TargetDevice>() { return cudaMemcpyDeviceToDevice;}

#ifdef HAVE_THRUST
template<typename T, typename Target> struct ThrustType;
template<typename T> struct ThrustType<T,TargetHost> { typedef T* Ptr; };
template<typename T> struct ThrustType<T,TargetDevice> { typedef thrust::device_ptr<T> Ptr; };
#endif // HAVE_THRUST

struct Manage
{
    inline static __host__
    void AllocateCheck()
    {
    }

    inline static __host__ __device__
    void AssignmentCheck()
    {
        assert(0);
        exit(-1);
    }

    template<typename T, typename Target> inline static __host__
    void Cleanup(T* ptr)
    {
        if(ptr) {
            Target::template DeallocatePitchedMem<T>(ptr);
        }
    }
};

struct DontManage
{
    inline static __host__
    void AllocateCheck()
    {
        std::cerr << "Image that doesn't own data should not call this constructor" << std::endl;
        assert(0);
        exit(-1);
    }

    inline static __host__ __device__
    void AssignmentCheck()
    {
    }


    template<typename T, typename Target> inline static __device__ __host__
    void Cleanup(T* ptr)
    {
    }
};

//! Return v clamped to interval [vmin,vmax]
template<typename T> __host__ __device__
inline T clamp(T vmin, T vmax, T v) {
    return v < vmin ? vmin : (vmax < v ? vmax : v);
}


//! Simple templated strided image type for use with Cuda
//! Type encapsulates ptr, pitch, stride, width and height
//! Instantiate Image<T,Target,ManagementAllocDealloc> to handle memory allocation
template<typename T, typename Target = TargetDevice, typename Management = DontManage>
struct Image {

    inline __device__ __host__
    ~Image()
    {
        Management::template Cleanup<T,Target>(ptr);
    }

    template<typename ManagementCopyFrom> inline __host__ __device__
    Image( const Image<T,Target,ManagementCopyFrom>& img )
        : ptr(img.ptr), pitch(img.pitch), stride(img.stride), w(img.w), h(img.h)
    {
        Management::AssignmentCheck();
    }

    inline __host__
    Image()
        :w(0), h(0), pitch(0), stride(0), ptr(0)
    {
    }

    inline __host__
    Image(unsigned int w, unsigned int h)
        :w(w), h(h)
    {
        Management::AllocateCheck();
        Target::template AllocatePitchedMem<T>(&ptr,&pitch,w,h);
        stride = pitch / sizeof(T);
    }

    inline __device__ __host__
    Image(T* ptr)
        :ptr(ptr), pitch(0), stride(0), w(0), h(0)
    {
    }

    inline __device__ __host__
    Image(T* ptr, size_t w)
        :ptr(ptr), pitch(sizeof(T)*w), stride(w), w(w), h(0)
    {
    }

    inline __device__ __host__
    Image(T* ptr, size_t w, size_t h)
        :ptr(ptr), pitch(sizeof(T)*w), stride(w), w(w), h(h)
    {
    }

    inline __device__ __host__
    Image(T* ptr, size_t w, size_t h, size_t stride)
        :ptr(ptr), pitch(sizeof(T)*stride), stride(stride), w(w), h(h)
    {
    }

    template<typename TargetFrom, typename ManagementFrom>
    inline __host__
    void CopyFrom(const Image<T,TargetFrom,ManagementFrom>& img)
    {
        cudaMemcpy2D(ptr,pitch,img.ptr,img.pitch, std::min(img.w,w)*sizeof(T), std::min(img.h,h), TargetCopyKind<Target,TargetFrom>() );
    }

    inline  __device__ __host__
    T& operator()(size_t x, size_t y)
    {
        return ptr[y*stride + x];
    }

    inline  __device__ __host__
    T& operator[](size_t ix)
    {
        return ptr[ix];
    }

    inline  __device__ __host__
    const T& operator()(size_t x, size_t y) const
    {
        return ptr[y*stride + x];
    }

    inline  __device__ __host__
    const T& operator[](size_t ix) const
    {
        return ptr[ix];
    }

    inline  __device__ __host__
    const T& GetWithClampedRange(int x, int y) const
    {
        x = Gpu::clamp<int>(0,w-1,x);
        y = Gpu::clamp<int>(0,h-1,y);
        return ptr[y*stride + x];
    }

    inline  __device__ __host__
    bool InBounds(int x, int y) const
    {
        return 0 <= x && x < w && 0 <= y && y < h;
    }

    template <typename DT>
    inline __host__
    void MemcpyFromHost(DT* hptr, size_t hpitch )
    {
        cudaMemcpy2D( (void*)ptr, pitch, hptr, hpitch, w*sizeof(T), h, cudaMemcpyHostToDevice );
    }

    template <typename DT>
    inline __host__
    void MemcpyFromHost(DT* ptr )
    {
        MemcpyFromHost(ptr, w*sizeof(T) );
    }

    inline __device__ __host__
    Image<T,Target,DontManage> SubImage(int x, int y, int width, int height) const
    {
        assert( (x+width) <= w && (y+height) <= h);
        return Image<T,Target,DontManage>(ptr + y*stride + x, width, height, stride);
    }

    inline __device__ __host__
    Image<T,Target,DontManage> Row(int y) const
    {
        return SubImage(0,y,w,1);
    }

    inline __device__ __host__
    Image<T,Target,DontManage> Col(int x) const
    {
        return SubImage(x,0,1,h);
    }

    inline __device__ __host__
    Image<T,Target,DontManage> SubImage(int width, int height)
    {
        assert(width <= w && height <= h);
        return Image<T,Target,DontManage>(ptr, width, height, stride);
    }

    //! Ignore this images stride - just return new image of
    //! size w x h which uses this memory
    template<typename TP>
    inline __device__ __host__
    Image<TP,Target,DontManage> PackedImage(int width, int height)
    {
        assert(width*height*sizeof(TP) <= w*h*sizeof(T) );
        return Image<TP,Target,DontManage>((TP*)ptr, width, height, width);
    }

#ifdef HAVE_NPP
    inline __device__ __host__
    Image<T,Target,DontManage> SubImage(const NppiRect& region)
    {
        return Image<T,Target,DontManage>(&(this->operator ()(region.x,region.y)), region.width, region.height, stride);
    }

    inline __device__ __host__
    Image<T,Target,DontManage> SubImage(const NppiSize& size)
    {
        return Image<T,Target,DontManage>(ptr, size.width,size.height, stride);
    }

    inline __host__
    const NppiSize Size() const
    {
        NppiSize ret = {(int)w,(int)h};
        return ret;
    }

    inline __host__
    const NppiRect Rect() const
    {
        NppiRect ret = {0,0,w,h};
        return ret;
    }
#endif

#ifdef HAVE_THRUST
    inline __device__ __host__
    typename Gpu::ThrustType<T,Target>::Ptr begin() {
        return (typename Gpu::ThrustType<T,Target>::Ptr)(ptr);
    }

    inline __device__ __host__
    typename Gpu::ThrustType<T,Target>::Ptr end() {
        return (typename Gpu::ThrustType<T,Target>::Ptr)(ptr + (h*stride) );
    }

    inline __host__
    void Fill(T val) {
        thrust::fill(begin(), end(), val);
    }

#endif

    T* ptr;
    size_t pitch;
    size_t stride;
    size_t w;
    size_t h;
};

}

#endif // CUDAIMAGE_H

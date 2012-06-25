#include <iostream>
#include <assert.h>
#include <boost/static_assert.hpp>

#include <cuda_runtime.h>

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
    Image(uint w, uint h)
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
        x = max((int)0, min((int)w-1,x));
        y = max((int)0, min((int)h-1,y));
        return ptr[y*stride + x];
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

    T* ptr;
    size_t pitch;
    size_t stride;
    size_t w;
    size_t h;
};

}

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>

#define HAVE_OPENCV

#ifndef __CUDACC__
    #ifdef HAVE_OPENCV
    #define USE_OPENCV
    #endif // HAVE_OPENCV
#endif // __CUDACC__

#ifdef USE_OPENCV
#include <opencv.hpp>
#endif // HAVE_OPENCV

#define HAVE_THRUST
#ifdef HAVE_THRUST
#include <thrust/device_vector.h>
#endif // HAVE_THRUST

#define HAVE_NPP
#ifdef HAVE_NPP
#include <npp.h>
#endif // HAVE_NPP

#include "Mat.h"
#include "sampling.h"
#include "pixel_convert.h"

namespace Gpu
{

struct CudaException : public std::exception
{
    CudaException(const std::string& what) : mWhat(what) { }
    virtual ~CudaException() throw() {}
    virtual const char* what() const throw() { return mWhat.c_str(); }
    std::string mWhat;
};

struct TargetHost
{
    template<typename T> inline static
    void AllocatePitchedMem(T** hostPtr, size_t *pitch, size_t w, size_t h){
        *pitch = w*sizeof(T);
        *hostPtr = (T*)malloc(*pitch * h);
    }

    template<typename T> inline static
    void AllocatePitchedMem(T** hostPtr, size_t *pitch, size_t *img_pitch, size_t w, size_t h, size_t d){
        *pitch = w*sizeof(T);
        *img_pitch = *pitch*h;
        *hostPtr = (T*)malloc(*pitch * h * d);
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
        if( cudaMallocPitch(devPtr,pitch,w*sizeof(T),h) != cudaSuccess ) {
            throw CudaException("Unable to cudaMallocPitch");
        }
    }

    template<typename T> inline static
    void AllocatePitchedMem(T** devPtr, size_t *pitch, size_t *img_pitch, size_t w, size_t h, size_t d){
        if( cudaMallocPitch(devPtr,pitch,w*sizeof(T),h*d) != cudaSuccess ) {
            throw CudaException("Unable to cudaMallocPitch");
        }
        *img_pitch = *pitch * h;
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
            ptr = 0;
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

//! Simple templated pitched image type for use with Cuda
//! Type encapsulates ptr, pitch, width and height
//! Instantiate Image<T,Target,ManagementAllocDealloc> to handle memory allocation
//! This struct is compatible with cudaPitchedPtr
template<typename T, typename Target = TargetDevice, typename Management = DontManage>
struct Image {

    inline __device__ __host__
    ~Image()
    {
        Management::template Cleanup<T,Target>(ptr);
    }

    //////////////////////////////////////////////////////
    // Constructors
    //////////////////////////////////////////////////////

    inline __host__ __device__
    Image( const Image<T,Target,Management>& img )
        : pitch(img.pitch), ptr(img.ptr), w(img.w), h(img.h)
    {
        Management::AssignmentCheck();
    }

    template<typename ManagementCopyFrom> inline __host__ __device__
    Image( const Image<T,Target,ManagementCopyFrom>& img )
        : pitch(img.pitch), ptr(img.ptr), w(img.w), h(img.h)
    {
        Management::AssignmentCheck();
    }

    inline __host__
    Image()
        : pitch(0), ptr(0), w(0), h(0)
    {
    }

    inline __host__
    Image(unsigned int w, unsigned int h)
        :w(w), h(h)
    {
        Management::AllocateCheck();
        Target::template AllocatePitchedMem<T>(&ptr,&pitch,w,h);
    }

    inline __device__ __host__
    Image(T* ptr)
        : pitch(0), ptr(ptr), w(0), h(0)
    {
    }

    inline __device__ __host__
    Image(T* ptr, size_t w)
        : pitch(sizeof(T)*w), ptr(ptr), w(w), h(0)
    {
    }

    inline __device__ __host__
    Image(T* ptr, size_t w, size_t h)
        : pitch(sizeof(T)*w), ptr(ptr), w(w), h(h)
    {
    }

    inline __device__ __host__
    Image(T* ptr, size_t w, size_t h, size_t pitch)
        : pitch(pitch), ptr(ptr), w(w), h(h)
    {
    }

#ifdef USE_OPENCV
    inline __host__
    Image( const cv::Mat& img )
        : pitch(img.step), ptr((T*)img.data), w(img.cols), h(img.rows)
    {
        // TODO: Assert only TargetHost
        Management::AssignmentCheck();
    }
#endif

#if __cplusplus > 199711L
    //////////////////////////////////////////////////////
    // R-Value Move assignment
    //////////////////////////////////////////////////////

    inline __host__
    Image(Image<T,Target,Management>&& img)
        : pitch(img.pitch), ptr(img.ptr), w(img.w), h(img.h)
    {
        // This object will take over managing data (if Management = Manage)
        img.ptr = 0;
    }

    inline __host__
    void operator=(Image<T,Target,Management>&& img)
    {
        assert(ptr==0);
        pitch = img.pitch;
        ptr = img.ptr;
        w = img.w;
        h = img.h;
        img.ptr = 0;
    }
#endif

    //////////////////////////////////////////////////////
    // Query dimensions
    //////////////////////////////////////////////////////

    inline __device__ __host__
    size_t Width() const
    {
        return w;
    }

    inline __device__ __host__
    size_t Height() const
    {
        return h;
    }

    inline __device__ __host__
    size_t Area() const
    {
        return w*h;
    }

    //////////////////////////////////////////////////////
    // Image set / copy
    //////////////////////////////////////////////////////

    inline __host__
    void Memset(unsigned char v = 0)
    {
        cudaMemset(ptr,v,pitch*h);
    }

    template<typename TargetFrom, typename ManagementFrom>
    inline __host__
    void CopyFrom(const Image<T,TargetFrom,ManagementFrom>& img)
    {
        cudaMemcpy2D(ptr,pitch,img.ptr,img.pitch, std::min(img.w,w)*sizeof(T), std::min(img.h,h), TargetCopyKind<Target,TargetFrom>() );
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

    template <typename DT>
    inline __host__
    void MemcpyToHost(DT* hptr, size_t hpitch )
    {
        cudaMemcpy2D( hptr, hpitch, (void*)ptr, pitch, w*sizeof(T), h, cudaMemcpyDeviceToHost );
    }

    template <typename DT>
    inline __host__
    void MemcpyToHost(DT* ptr )
    {
        MemcpyToHost(ptr, w*sizeof(T) );
    }


    inline __host__ __device__
    void Swap(Image<T,Target,Management>& img)
    {
        std::swap(img.pitch, pitch);
        std::swap(img.ptr, ptr);
        std::swap(img.w, w);
        std::swap(img.h, h);
    }

    //////////////////////////////////////////////////////
    // Direct Pixel Access
    //////////////////////////////////////////////////////

    inline __device__ __host__
    bool IsValid() const
    {
        return ptr != 0;
    }

    inline  __device__ __host__
    T* RowPtr(size_t y)
    {
        return (T*)((unsigned char*)(ptr) + y*pitch);
    }

    inline  __device__ __host__
    const T* RowPtr(size_t y) const
    {
        return (T*)((unsigned char*)(ptr) + y*pitch);
    }

    inline  __device__ __host__
    T& operator()(size_t x, size_t y)
    {
        return RowPtr(y)[x];
    }

    inline  __device__ __host__
    const T& operator()(size_t x, size_t y) const
    {
        return RowPtr(y)[x];
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
    const T& Get(int x, int y) const
    {
        return RowPtr(y)[x];
    }

    //////////////////////////////////////////////////////
    // Bounds Checking
    //////////////////////////////////////////////////////

    inline  __device__ __host__
    bool InBounds(int x, int y) const
    {
        return 0 <= x && x < w && 0 <= y && y < h;
    }

    inline  __device__ __host__
    bool InBounds(float x, float y, float border) const
    {
        return border <= x && x < (w-border) && border <= y && y < (h-border);
    }

    //////////////////////////////////////////////////////
    // Clamped / Interpolated access
    //////////////////////////////////////////////////////

    inline  __device__ __host__
    const T& GetWithClampedRange(int x, int y) const
    {
        x = clamp(x, 0, w-1);
        y = clamp(y, 0, h-1);
        return RowPtr(y)[x];
    }

    inline  __device__ __host__
    const T& GetConditionNeumann(int x, int y) const
    {
        x = abs(x);
        if(x >= w) x = (w-1)-(x-w);

        y = abs(y);
        if(y >= h) y = (h-1)-(y-h);

        return RowPtr(y)[x];
    }

    template<typename TR>
    inline __device__ __host__
    TR GetBilinear(float u, float v) const
    {
        const float ix = floorf(u);
        const float iy = floorf(v);
        const float fx = u - ix;
        const float fy = v - iy;

        const T* bl = RowPtr(iy)  + (size_t)ix;
        const T* tl = RowPtr(iy+1)+ (size_t)ix;

        return lerp(
            lerp( bl[0], bl[1], fx ),
            lerp( tl[0], tl[1], fx ),
            fy
        );
    }

    inline __device__ __host__
    T GetNearestNeighbour(float u, float v) const
    {
        return Get(u+0.5, v+0.5);
    }

    //////////////////////////////////////////////////////
    // Backward difference
    //////////////////////////////////////////////////////

    template<typename TR>
    inline __device__ __host__
    TR GetBackwardDiffDx(int x, int y) const
    {
        const T* row = RowPtr(y);
        return ( ConvertPixel<TR,T>(row[x]) - ConvertPixel<TR,T>(row[x-1]) );
    }

    template<typename TR>
    inline __device__ __host__
    TR GetBackwardDiffDy(int x, int y) const
    {
        return ( ConvertPixel<TR,T>(Get(x,y)) - ConvertPixel<TR,T>(Get(x,y-1)) );
    }

    //////////////////////////////////////////////////////
    // Central difference
    //////////////////////////////////////////////////////

    template<typename TR>
    inline __device__ __host__
    TR GetCentralDiffDx(int x, int y) const
    {
        const T* row = RowPtr(y);
//        return ((TR)row[x+1] - (TR)row[x-1]) / (TR)2;
        return ( ConvertPixel<TR,T>(row[x+1]) - ConvertPixel<TR,T>(row[x-1]) ) / 2;
    }

    template<typename TR>
    inline __device__ __host__
    TR GetCentralDiffDy(int x, int y) const
    {
        return ( ConvertPixel<TR,T>(Get(x,y+1)) - ConvertPixel<TR,T>(Get(x,y-1)) ) / 2;
    }

    template<typename TR>
    inline __device__ __host__
    Mat<TR,1,2> GetCentralDiff(int px, int py) const
    {
        Mat<TR,1,2> res;
        res(0) = GetCentralDiffDx<TR>(px,py);
        res(1) = GetCentralDiffDy<TR>(px,py);
        return res;
    }

    template<typename TR>
    inline __device__ __host__
    Mat<TR,1,2> GetCentralDiff(float px, float py) const
    {
        // TODO: Make more efficient by expanding GetCentralDiff calls
        const int ix = floor(px);
        const int iy = floor(py);
        const float fx = px - ix;
        const float fy = py - iy;

        const int b = py;   const int l = px;
        const int t = py+1; const int r = px+1;

        TR tldx = GetCentralDiffDx<TR>(l,t);
        TR trdx = GetCentralDiffDx<TR>(r,t);
        TR bldx = GetCentralDiffDx<TR>(l,b);
        TR brdx = GetCentralDiffDx<TR>(r,b);
        TR tldy = GetCentralDiffDy<TR>(l,t);
        TR trdy = GetCentralDiffDy<TR>(r,t);
        TR bldy = GetCentralDiffDy<TR>(l,b);
        TR brdy = GetCentralDiffDy<TR>(r,b);

        Mat<TR,1,2> res;
        res(0) = lerp(lerp(bldx,brdx,fx), lerp(tldx,trdx,fx), fy);
        res(1) = lerp(lerp(bldy,brdy,fx), lerp(tldy,trdy,fx), fy);
        return res;
    }

    //////////////////////////////////////////////////////
    // Obtain slices / subimages
    //////////////////////////////////////////////////////

    inline __device__ __host__
    const Image<T,Target,DontManage> SubImage(int x, int y, int width, int height) const
    {
        assert( (x+width) <= w && (y+height) <= h);
        return Image<T,Target,DontManage>( RowPtr(y)+x, width, height, pitch);
    }

    inline __device__ __host__
    Image<T,Target,DontManage> SubImage(int x, int y, int width, int height)
    {
        assert( (x+width) <= w && (y+height) <= h);
        return Image<T,Target,DontManage>( RowPtr(y)+x, width, height, pitch);
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
        return Image<T,Target,DontManage>(ptr, width, height, pitch);
    }

    //////////////////////////////////////////////////////
    // Reuse image memory buffer for different images
    //////////////////////////////////////////////////////

    //! Ignore this images pitch - just return new image of
    //! size w x h which uses this memory
    template<typename TP>
    inline __device__ __host__
    Image<TP,Target,DontManage> PackedImage(int width, int height)
    {
        assert(width*height*sizeof(TP) <= h*pitch );
        return Image<TP,Target,DontManage>((TP*)ptr, width, height, width*sizeof(TP) );
    }

    template<typename TP>
    inline __device__ __host__
    Image<TP,Target,DontManage> AlignedImage(int width, int height, int align_bytes=16)
    {
        const int wbytes = width*sizeof(TP);
        const int npitch = (wbytes%align_bytes) == 0 ? wbytes : align_bytes*(1 + wbytes/align_bytes);
        assert(npitch*height <= h*pitch );
        return Image<TP,Target,DontManage>((TP*)ptr, width, height, npitch );
    }

    //! Split image into smaller typed image and remaining space
    //! Use for simple CPU/GPU memory pool
    //! Only applicable for DontManage types
    template<typename TP>
    inline __device__ __host__
    Image<TP,Target,DontManage> SplitAlignedImage(unsigned int nwidth, unsigned int nheight, unsigned int align_bytes=16)
    {
        // Only let us split DontManage image types (so we can't orthan memory)
        Management::AssignmentCheck();

        // Extract aligned image of type TP from start of this image
        const unsigned int wbytes = nwidth*sizeof(TP);
        const unsigned int npitch = (wbytes%align_bytes) == 0 ? wbytes : align_bytes*(1 + wbytes/align_bytes);
        TP* nptr = (TP*)ptr;
        assert(npitch*nheight <= h*pitch );

        // Update this image to reflect remainder (as single row image)
        ptr = (T*)((unsigned char*)(ptr) + nheight*npitch);
        pitch = (h*pitch - nheight*npitch);
        h = 1;
        w = pitch / sizeof(T);

        return Image<TP,Target,DontManage>(nptr, nwidth, nheight, npitch );
    }

    //////////////////////////////////////////////////////
    // Accessors for cuda vector types
    //////////////////////////////////////////////////////

    inline __device__ __host__
    T GetNearestNeighbour(const float2 p) const
    {
        return GetNearestNeighbour(p.x,p.y);
    }

    template<typename TR>
    inline __device__ __host__
    TR GetBilinear(const float2 p) const
    {
        return GetBilinear<TR>(p.x, p.y);
    }

    inline  __device__ __host__
    bool InBounds(const float2 p, float border) const
    {
        return InBounds(p.x, p.y, border);
    }


    //////////////////////////////////////////////////////
    // NVidia Performance Primitives convenience methods
    //////////////////////////////////////////////////////

#ifdef HAVE_NPP
    inline __device__ __host__
    Image<T,Target,DontManage> SubImage(const NppiRect& region)
    {
        return Image<T,Target,DontManage>(RowPtr(region.y)+region.x, region.width, region.height, pitch);
    }

    inline __device__ __host__
    Image<T,Target,DontManage> SubImage(const NppiSize& size)
    {
        return Image<T,Target,DontManage>(ptr, size.width,size.height, pitch);
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

    //////////////////////////////////////////////////////
    // Thrust convenience methods
    //////////////////////////////////////////////////////

#ifdef HAVE_THRUST
    inline __device__ __host__
    typename Gpu::ThrustType<T,Target>::Ptr begin() {
        return (typename Gpu::ThrustType<T,Target>::Ptr)(ptr);
    }

    inline __device__ __host__
    typename Gpu::ThrustType<T,Target>::Ptr end() {
        return (typename Gpu::ThrustType<T,Target>::Ptr)( RowPtr(h-1) + w );
    }

    inline __host__
    void Fill(T val) {
        thrust::fill(begin(), end(), val);
    }

#endif

    //////////////////////////////////////////////////////
    // Member variables
    //////////////////////////////////////////////////////

    size_t pitch;
    T* ptr;
    size_t w;
    size_t h;
};

}

#endif // CUDAIMAGE_H

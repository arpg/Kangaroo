#pragma once

#include <kangaroo/Image.h>

namespace roo
{

// Power of two pyramid.
template<typename T, unsigned Levels, typename Target = TargetDevice, typename Management = DontManage>
struct Pyramid {

    inline __device__ __host__
    ~Pyramid()
    {
        // Each image layer will clean up itself
    }

    //////////////////////////////////////////////////////
    // Constructors
    //////////////////////////////////////////////////////

    inline __host__
    Pyramid(unsigned w, unsigned h)
    {
        // Check this Pyramid supports memory management
        Management::AllocateCheck();

        // Build power of two structure
        for(unsigned l=0; l < Levels && (w>>l > 0) && (h>>l > 0); ++l ) {
            // Avoid r-value assignment in case it isn't available.
            roo::Image<T,Target,Management> temp(w>>l,h>>l);
            imgs[l].Swap(temp);
        }
    }

    template<typename TargetFrom, typename ManagementFrom>
    inline __host__ __device__
    Pyramid(const Pyramid<T,Levels,TargetFrom,ManagementFrom>& pyramid)
    {
        AssignmentCheck<Management,Target,TargetFrom>();
        for(unsigned int l=0; l<Levels; ++l) {
            imgs[l] = pyramid.imgs[l];
        }
    }

    inline __host__
    Pyramid()
    {
        // Unassigned layers
    }

    //////////////////////////////////////////////////////
    // Pyramid set / copy
    //////////////////////////////////////////////////////

    inline __host__
    void Memset(unsigned char v = 0)
    {
        for(int l=0; l<Levels; ++l) {
            imgs[l].Memset(v);
        }
    }

    template<typename TargetFrom, typename ManagementFrom>
    inline __host__
    void CopyFrom(const Pyramid<T,Levels,TargetFrom,ManagementFrom>& pyramid)
    {
        for(unsigned int l=0; l<Levels; ++l) {
            imgs[l].CopyFrom(pyramid.imgs[l]);
        }
    }

    inline __host__ __device__
    void Swap(Pyramid<T,Levels,Target,Management>& pyramid)
    {
        for(int l=0; l<Levels; ++l) {
            imgs[l].Swap(pyramid.imgs[l]);
        }
    }

    //////////////////////////////////////////////////////
    // Image accessors
    //////////////////////////////////////////////////////

    inline __host__ __device__
    roo::Image<T,Target,Management>& operator[](size_t i)
    {
        return imgs[i];
    }

    inline __host__ __device__
    const roo::Image<T,Target,Management>& operator[](size_t i) const
    {
        return imgs[i];
    }

    //////////////////////////////////////////////////////
    // Obtain slices / subimages
    //////////////////////////////////////////////////////

    template<unsigned SubLevels>
    inline __host__
    Pyramid<T,SubLevels,Target,DontManage> SubPyramid(unsigned startLevel)
    {
        assert(startLevel + SubLevels <= Levels);
        Pyramid<T,SubLevels,Target,DontManage> pyr;

        for(unsigned l=0; l < SubLevels; ++l) {
            pyr.imgs[l] = imgs[startLevel+l];
        }

        return pyr;
    }

    //////////////////////////////////////////////////////
    // Allocate from image (as memory pool)
    //////////////////////////////////////////////////////

    inline __host__
    void AllocateFromImage(unsigned w, unsigned h, Image<unsigned char, Target, DontManage> scratch )
    {
        // Verify that this is DontManage pyramid type
        AssignmentCheck<Management,Target,Target>();

        // Build power of two structure
        for(unsigned l=0; l < Levels && (w>>l > 0) && (h>>l > 0); ++l ) {
            imgs[l] = scratch.template SplitAlignedImage<T>(w>>l,h>>l);
        }
    }


    //////////////////////////////////////////////////////
    // Member variables
    //////////////////////////////////////////////////////

    roo::Image<T,Target,Management> imgs[Levels];
};


}

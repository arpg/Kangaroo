#pragma once

#include "Image.h"

namespace Gpu
{

// Power of two pyramid.
template<typename T, unsigned Levels, typename Target = TargetDevice, typename Management = DontManage>
struct Pyramid {

    //////////////////////////////////////////////////////
    // Constructors
    //////////////////////////////////////////////////////

    inline __host__
    Pyramid(unsigned w, unsigned h)
    {
        // Check this Pyramid supports memory management
        Management::AllocateCheck();

        // Build power of two structure
        for(unsigned l=0; l < Levels && (w>>l > 0) && (h>>l > 0); ++l )
        {
            imgs[l] = Gpu::Image<T,Target,Management>(w>>l,h>>l);
        }
    }

    template<typename ManagementCopyFrom>
    inline __host__ __device__
    Pyramid(const Pyramid<T,Levels,Target,ManagementCopyFrom>& pyramid)
    {
        Management::AssignmentCheck();
        for(int l=0; l<Levels; ++l) {
            imgs[l] = pyramid.imgs[l];
        }
    }

    //////////////////////////////////////////////////////
    // Image accessors
    //////////////////////////////////////////////////////

    Gpu::Image<T,Target,Management>& operator[](size_t i)
    {
        return imgs[i];
    }

    const Gpu::Image<T,Target,Management>& operator[](size_t i) const
    {
        return imgs[i];
    }

    //////////////////////////////////////////////////////
    // Member variables
    //////////////////////////////////////////////////////

    Gpu::Image<T,Target,Management> imgs[Levels];
};


}

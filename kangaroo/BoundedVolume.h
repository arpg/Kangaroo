#pragma once

#include "Volume.h"
#include "BoundingBox.h"

namespace Gpu
{

template<typename T, typename Target = TargetDevice, typename Management = DontManage>
class BoundedVolume : public Volume<T,Target,Management>
{
public:

    //////////////////////////////////////////////////////
    // Constructors
    //////////////////////////////////////////////////////

    template<typename ManagementCopyFrom> inline __host__ __device__
    BoundedVolume( const BoundedVolume<T,Target,ManagementCopyFrom>& img )
        : Volume<T,Target,Management>(img), bbox(img.bbox)
    {
        Management::AssignmentCheck();
    }

    inline __host__
    BoundedVolume()
    {
    }

    inline __host__
    BoundedVolume(unsigned int w, unsigned int h, unsigned int d)
        : Volume<T,Target,Management>(w,h,d),
          bbox(make_float3(-1,-1,-1), make_float3(1,1,1))
    {
    }

    inline __host__
    BoundedVolume(unsigned int w, unsigned int h, unsigned int d, float3 min_bounds, float3 max_bounds)
        : Volume<T,Target,Management>(w,h,d),
          bbox(min_bounds,max_bounds)
    {
    }

    //////////////////////////////////////////////////////
    // Dimensions
    //////////////////////////////////////////////////////

    inline __device__ __host__
    float3 SizeUnits() const
    {
        return bbox.Size();
    }

    inline __device__ __host__
    float3 VoxelSizeUnits() const
    {
        return bbox.Size() /
            make_float3(
                Volume<T,Target,Management>::w-1,
                Volume<T,Target,Management>::h-1,
                Volume<T,Target,Management>::d-1
            );
    }

    //////////////////////////////////////////////////////
    // Access volume in units of Bounding Box
    //////////////////////////////////////////////////////

    inline  __device__ __host__
    float GetUnitsTrilinearClamped(float3 pos_w) const
    {
        const float3 pos_v = (pos_w - bbox.Min()) / (bbox.Max() - bbox.Min());
        return Volume<T,Target,Management>::GetFractionalTrilinearClamped(pos_v);
    }

    inline __device__ __host__
    float3 GetUnitsBackwardDiffDxDyDz(float3 pos_w) const
    {
        const float3 pos_v = (pos_w - bbox.Min()) / (bbox.Max() - bbox.Min());
        return Volume<T,Target,Management>::GetFractionalBackwardDiffDxDyDz(pos_v);
    }

    inline __device__ __host__
    float3 VoxelPositionInUnits(int x, int y, int z)
    {
        const float3 vol_size = bbox.Size();

        return make_float3(
            bbox.Min().x + vol_size.x*x/(float)(Volume<T,Target,Management>::w-1),
            bbox.Min().y + vol_size.y*y/(float)(Volume<T,Target,Management>::h-1),
            bbox.Min().z + vol_size.z*z/(float)(Volume<T,Target,Management>::d-1)
        );
    }

    BoundingBox bbox;
};

}

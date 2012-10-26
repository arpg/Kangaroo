#pragma once

#include "Mat.h"

namespace Gpu
{

// Axis Aligned Bounding Box
struct BoundingBox
{
    inline __device__ __host__
    BoundingBox()
    {
    }

    inline __device__ __host__
    BoundingBox(float3 boxmin, float3 boxmax)
        : boxmin(boxmin), boxmax(boxmax)
    {
    }

    // Construct bounding box from Frustum.
    inline __host__
    BoundingBox(
        const Mat<float,3,4> T_wc,
        float w, float h,
        float fu, float fv, float u0, float v0,
        float near, float far
    ) {
        Clear();

        // Insert each edge of frustum into bounding box
        const float3 c_w = SE3Translation(T_wc);
        const float3 ray_tl = mulSO3(T_wc, make_float3((0-u0)/fu,(0-v0)/fv, 1));
        const float3 ray_tr = mulSO3(T_wc, make_float3((w-u0)/fu,(0-v0)/fv, 1));
        const float3 ray_bl = mulSO3(T_wc, make_float3((0-u0)/fu,(h-v0)/fv, 1));
        const float3 ray_br = mulSO3(T_wc, make_float3((w-u0)/fu,(h-v0)/fv, 1));

        Insert(c_w + near*ray_tl);
        Insert(c_w + near*ray_tr);
        Insert(c_w + near*ray_bl);
        Insert(c_w + near*ray_br);
        Insert(c_w + far*ray_tl);
        Insert(c_w + far*ray_tr);
        Insert(c_w + far*ray_bl);
        Insert(c_w + far*ray_br);
    }

    inline __host__ __device__
    float3& Min() {
        return boxmin;
    }

    inline __host__ __device__
    float3 Min() const {
        return boxmin;
    }

    inline __host__ __device__
    float3& Max() {
        return boxmax;
    }

    inline __host__ __device__
    float3 Max() const {
        return boxmax;
    }


    inline __host__
    void Clear()
    {
        boxmin = make_float3(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max());
        boxmax = make_float3(std::numeric_limits<float>::min(),std::numeric_limits<float>::min(),std::numeric_limits<float>::min());
    }

    // Expand bounding box to include p
    inline __host__
    void Insert(float3 p)
    {
        boxmin = make_float3(fminf(p.x,boxmin.x), fminf(p.x,boxmin.y), fminf(p.x,boxmin.z));
        boxmax = make_float3(fmaxf(p.x,boxmax.x), fmaxf(p.x,boxmax.y), fmaxf(p.x,boxmax.z));
    }

    // Expand bounding box to include bb
    inline __host__
    void Insert(BoundingBox bb)
    {
        boxmin = make_float3(fminf(bb.boxmin.x,boxmin.x), fminf(bb.boxmin.x,boxmin.y), fminf(bb.boxmin.x,boxmin.z));
        boxmax = make_float3(fmaxf(bb.boxmax.x,boxmax.x), fmaxf(bb.boxmax.x,boxmax.y), fmaxf(bb.boxmax.x,boxmax.z));
    }

    // Contract bounding box to represent intersection (common space)
    // between this and bb
    inline __host__
    void Intersect(BoundingBox bb)
    {
        boxmin = make_float3(fmaxf(bb.boxmin.x,boxmin.x), fmaxf(bb.boxmin.x,boxmin.y), fmaxf(bb.boxmin.x,boxmin.z));
        boxmax = make_float3(fminf(bb.boxmax.x,boxmax.x), fminf(bb.boxmax.x,boxmax.y), fminf(bb.boxmax.x,boxmax.z));
    }

    inline __host__ __device__
    float3 Size() const
    {
        return boxmax - boxmin;
    }

    float3 boxmin;
    float3 boxmax;
};

}

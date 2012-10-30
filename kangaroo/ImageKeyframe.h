#pragma once

#include "Image.h"
#include "MatUtils.h"

namespace Gpu
{

struct ImageIntrinsics
{

    //////////////////////////////////////////////////////
    // Constructors
    //////////////////////////////////////////////////////

    inline __host__ __device__
    ImageIntrinsics()
        : fu(0), fv(0), u0(0), v0(0)
    {
    }

    inline __host__ __device__
    ImageIntrinsics(float fu, float fv, float u0, float v0)
        : fu(fu), fv(fv), u0(u0), v0(v0)
    {
    }

    inline __host__ __device__
    ImageIntrinsics(float f, float u0, float v0)
        : fu(f), fv(f), u0(u0), v0(v0)
    {
    }

    template<typename T, typename Target, typename Manage>
    inline __host__ __device__
    ImageIntrinsics(float f, const Image<T,Target,Manage>& img)
        : fu(f), fv(f), u0(img.w/2.0f - 0.5), v0(img.h/2.0f - 0.5)
    {
    }

    //////////////////////////////////////////////////////
    // Image projection
    //////////////////////////////////////////////////////

    inline __host__ __device__
    float2 Project(const float3 P_c) const
    {
        return make_float2(u0 + fu*P_c.x/P_c.z, v0 + fv*P_c.y/P_c.z);
    }

    inline __host__ __device__
    float2 Project(float x, float y, float z) const
    {
        return make_float2(u0 + fu*x/z, v0 + fv*y/z);
    }

    inline __host__ __device__
    float2 operator*(float3 P_c) const
    {
        return Project(P_c);
    }

    //////////////////////////////////////////////////////
    // Image Unprojection
    //////////////////////////////////////////////////////

    inline __host__ __device__
    float3 Unproject(float u, float v) const
    {
        return make_float3((u-u0)/fu,(v-v0)/fv, 1);
    }

    inline __host__ __device__
    float3 Unproject(const float2 p_c) const
    {
        return make_float3((p_c.x-u0)/fu,(p_c.y-v0)/fv, 1);
    }

    inline __host__ __device__
    float3 Unproject(const float2 p_c, float z) const
    {
        return z * Unproject(p_c);
    }

    inline __host__ __device__
    float3 Unproject(float u, float v, float z) const
    {
        return z * Unproject(u,v);
    }

    //////////////////////////////////////////////////////
    // Intrinsics for pow 2 pyramid
    //////////////////////////////////////////////////////

    inline __host__ __device__
    ImageIntrinsics operator[](int l) const
    {
        const float scale = 1.0f / (1 << l);
        return ImageIntrinsics(scale*fu, scale*fv, scale*(u0+0.5f)-0.5f, scale*(v0+0.5f)-0.5f);
    }

    float fu;
    float fv;
    float u0;
    float v0;
};

template<typename T, typename Target = TargetDevice, typename Management = DontManage>
struct ImageKeyframe
{
    inline __host__ __device__
    float2 Project(const float3 P_w) const
    {
        return K * (T_iw * P_w);
    }

    ImageIntrinsics K;
    Mat<float,3,4> T_iw;
    Image<T, Gpu::TargetDevice> img;
};

} // namespace Gpu

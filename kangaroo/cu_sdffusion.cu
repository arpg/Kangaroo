#include "MatUtils.h"
#include "Image.h"
#include "Volume.h"
#include "Sdf.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Truncated SDF Fusion
//////////////////////////////////////////////////////

__global__ void KernSdfFuse(Volume<SDF_t> vol, float3 vol_min, float3 vol_max, Image<float> depth, Mat<float,3,4> T_cw, float fu, float fv, float u0, float v0, float trunc_dist )
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z*blockDim.z + threadIdx.z;

    const float3 vol_size = vol_max - vol_min;

    const float3 P_w = make_float3(
                vol_min.x + vol_size.x*x/(float)(vol.w-1),
                vol_min.y + vol_size.y*y/(float)(vol.h-1),
                vol_min.z + vol_size.z*z/(float)(vol.d-1)
                );

    const float3 P_c = T_cw * P_w;
    const float2 p_c = make_float2(u0 + fu*P_c.x/P_c.z, v0 + fv*P_c.y/P_c.z);
    const int2 xy = make_int2(p_c.x+0.5, p_c.y + 0.5);

    if( 0 <= xy.x && xy.x < depth.w && 0 <= xy.y && xy.y < depth.h)
    {
        const float vd = length(P_c);
        const float md = depth.Get(xy.x, xy.y);
        const float sd = md - vd;

//        const float dist = length(P_w - make_float3(0,0,0));
//        vol(x,y,z) = SDF_t(dist - 0.95);

        if(sd < -trunc_dist) {
            // Further than truncation distance from surface
            // We do nothing.
        }else{
            const float val = fminf(1, sd / trunc_dist) * (sd / abs(sd));
            const SDF_t current = vol(x,y,z);
            vol(x,y,z) = SDF_t(val);
        }
    }
 }

void SdfFuse(Volume<SDF_t> vol, float3 vol_min, float3 vol_max, Image<float> depth, Mat<float,3,4> T_cw, float fu, float fv, float u0, float v0, float trunc_dist )
{
    dim3 blockDim(8,8,8);
    dim3 gridDim(vol.w / blockDim.x, vol.h / blockDim.y, vol.d / blockDim.z);
    KernSdfFuse<<<gridDim,blockDim>>>(vol, vol_min, vol_max, depth, T_cw, fu, fv, u0, v0, trunc_dist);
}

//////////////////////////////////////////////////////
// Reset SDF
//////////////////////////////////////////////////////

__global__ void KernSdfReset(Volume<SDF_t> vol, float trunc_dist)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z*blockDim.z + threadIdx.z;

    vol(x,y,z) = SDF_t(trunc_dist, 0);
}

void SdfReset(Volume<SDF_t> vol, float trunc_dist)
{
    dim3 blockDim(8,8,8);
    dim3 gridDim(vol.w / blockDim.x, vol.h / blockDim.y, vol.d / blockDim.z);
    KernSdfReset<<<gridDim,blockDim>>>(vol, trunc_dist);
}

//////////////////////////////////////////////////////
// Create SDF representation of sphere
//////////////////////////////////////////////////////

__global__ void KernSdfSphere(Volume<SDF_t> vol, float3 vol_min, float3 vol_max, float3 center, float r)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z*blockDim.z + threadIdx.z;

    const float3 vol_size = vol_max - vol_min;

    const float3 pos = make_float3(
                vol_min.x + vol_size.x*x/(float)(vol.w-1),
                vol_min.y + vol_size.y*y/(float)(vol.h-1),
                vol_min.z + vol_size.z*z/(float)(vol.d-1)
                );
    const float dist = length(pos - center);
    const float sdf = dist - r;

    vol(x,y,z) = SDF_t(sdf);
}

void SdfSphere(Volume<SDF_t> vol, float3 vol_min, float3 vol_max, float3 center, float r)
{
    dim3 blockDim(8,8,8);
    dim3 gridDim(vol.w / blockDim.x, vol.h / blockDim.y, vol.d / blockDim.z);

    KernSdfSphere<<<gridDim,blockDim>>>(vol, vol_min, vol_max, center, r);
}


}

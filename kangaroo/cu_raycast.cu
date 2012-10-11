#include "kangaroo.h"
#include "launch_utils.h"
#include "Mat.h"

namespace Gpu
{

__global__ void KernRaycast(Image<float> img, const Volume<SDF_t> vol, const float3 boxmin, const float3 boxmax, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float near, float far, bool subpix )
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    if( u < img.w && v < img.h ) {
        const float3 c_w = SE3Translation(T_wc);
        const float3 ray_c = make_float3((u-u0)/fu,(v-v0)/fv, 1);
        const float3 ray_w = mulSO3(T_wc, ray_c);

        // TODO: Ensure the bbox boundaries generate valid trilinear access

        // http://www.cs.utah.edu/~awilliam/box/box.pdf
        const float3 tminbound = (boxmin - c_w) / ray_w;
        const float3 tmaxbound = (boxmax - c_w) / ray_w;
        const float3 tmin = fminf(tminbound,tmaxbound);
        const float3 tmax = fmaxf(tminbound,tmaxbound);
        const float max_tmin = fmaxf(fmaxf(fmaxf(tmin.x, tmin.y), tmin.z), near);
        const float min_tmax = fminf(fminf(fminf(tmax.x, tmax.y), tmax.z), far);

        float ret = 0.0f;

        if(max_tmin < min_tmax ) {
//            // Raycast cube
//            ret = (max_tmin - near) / (far - near);

            // Go between max_tmin and min_tmax
            float lambda = max_tmin;
            float last_sdf = 0;
            const float delta_lambda = (boxmax.x - boxmin.x) / (4*vol.w);

            while(lambda < min_tmax) {
                const float3 pos_w = c_w + lambda * ray_w;
                const float3 pos_v = (pos_w - boxmin) / (boxmax - boxmin);
                const SDF_t val = vol.GetFractional(pos_v);
                const float sdf = val.val / val.n;
                if( sdf <= 0 ) {
                    // surface!
                    if(subpix) {
                        lambda = lambda - delta_lambda * last_sdf / (sdf - last_sdf);
                    }
                    ret = (lambda - near) / (far - near);
                    break;
                }
                lambda += delta_lambda;
                last_sdf = sdf;
            }
        }

        img(u,v) = ret;
    }
}

void Raycast(Image<float> img, const Volume<SDF_t> vol, const float3 boxmin, const float3 boxmax, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float near, float far, bool subpix )
{    
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim, gridDim, img);
    KernRaycast<<<gridDim,blockDim>>>(img, vol, boxmin, boxmax, T_wc, fu, fv, u0, v0, near, far, subpix);
}

__global__ void KernSDFSphere(Volume<SDF_t> vol, float3 vol_min, float3 vol_max, float3 center, float r)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z*blockDim.z + threadIdx.z;

    const float3 vol_size = vol_max - vol_min;

    const float3 pos = make_float3(
                vol_min.x + vol_size.x*x/(float)vol.w,
                vol_min.y + vol_size.y*y/(float)vol.h,
                vol_min.z + vol_size.z*z/(float)vol.d
                );
    const float dist = length(pos - center);
    const float sdf = dist - r;

    vol(x,y,z) = SDF_t(sdf);
}

void SDFSphere(Volume<SDF_t> vol, float3 vol_min, float3 vol_max, float3 center, float r)
{
    dim3 blockDim(8,8,8);
    dim3 gridDim(vol.w / blockDim.x, vol.h / blockDim.y, vol.d / blockDim.z);

    KernSDFSphere<<<gridDim,blockDim>>>(vol, vol_min, vol_max, center, r);
}


}

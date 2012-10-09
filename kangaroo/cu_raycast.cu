#include "kangaroo.h"
#include "launch_utils.h"
#include "Mat.h"

namespace Gpu
{

__global__ void KernRaycast(Image<float> img, const Volume<SDF_t> vol, const float3 boxmin, const float3 boxmax, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float near, float far )
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    if( u < img.w && v < img.h ) {
        const float3 c_w = SE3Translation(T_wc);
        const float3 ray_c = make_float3((u-u0)/fu,(v-v0)/fv, 1);
        const float3 ray_w = mulSO3(T_wc, ray_c);

        // http://www.cs.utah.edu/~awilliam/box/box.pdf
        const float3 tminbound = (boxmin - c_w) / ray_w;
        const float3 tmaxbound = (boxmax - c_w) / ray_w;
        const float3 tmin = fminf(tminbound,tmaxbound);
        const float3 tmax = fmaxf(tminbound,tmaxbound);
        const float max_tmin = fmaxf(fmaxf(fmaxf(tmin.x, tmin.y), tmin.z), near);
        const float min_tmax = fminf(fminf(fminf(tmax.x, tmax.y), tmax.z), far);

        float ret = 0.0f;

        if(max_tmin < min_tmax ) {
            ret = (max_tmin - near) / (far - near);
        }

        img(u,v) = ret;
    }
}

void Raycast(Image<float> img, const Volume<SDF_t> vol, const float3 boxmin, const float3 boxmax, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float near, float far )
{    
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim, gridDim, img);
    KernRaycast<<<gridDim,blockDim>>>(img, vol, boxmin, boxmax, T_wc, fu, fv, u0, v0, near, far);
}

void SDFSphere(Volume<SDF_t> vol, float3 xyz, float r)
{

}


}

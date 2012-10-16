#include "Mat.h"
#include "MatUtils.h"
#include "Image.h"
#include "Volume.h"
#include "Sdf.h"
#include "launch_utils.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Raycast SDF
//////////////////////////////////////////////////////

__global__ void KernRaycastSDF(Image<float> imgdepth, Image<float4> norm, Image<float> img, const Volume<SDF_t> vol, const float3 boxmin, const float3 boxmax, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float near, float far, float trunc_dist, bool subpix )
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    if( u < img.w && v < img.h ) {
        const float3 c_w = SE3Translation(T_wc);
        const float3 ray_c = make_float3((u-u0)/fu,(v-v0)/fv, 1);
        const float3 ray_w = mulSO3(T_wc, ray_c);

        // Raycast bounding box to find valid ray segment of sdf
        // http://www.cs.utah.edu/~awilliam/box/box.pdf
        const float3 tminbound = (boxmin - c_w) / ray_w;
        const float3 tmaxbound = (boxmax - c_w) / ray_w;
        const float3 tmin = fminf(tminbound,tmaxbound);
        const float3 tmax = fmaxf(tminbound,tmaxbound);
        const float max_tmin = fmaxf(fmaxf(fmaxf(tmin.x, tmin.y), tmin.z), near);
        const float min_tmax = fminf(fminf(fminf(tmax.x, tmax.y), tmax.z), far);

        float depth = 0.0f;

        // If ray intersects bounding box
        if(max_tmin < min_tmax ) {
            // Go between max_tmin and min_tmax
            float lambda = max_tmin;
            float last_sdf = 0;
            float delta_lambda = (boxmax.x - boxmin.x) / (vol.w-1);

            // March through space
            while(lambda < min_tmax) {
                const float3 pos_w = c_w + lambda * ray_w;
                const float3 pos_v = (pos_w - boxmin) / (boxmax - boxmin);
                const float sdf = vol.GetFractionalTrilinearClamped(pos_v);

                if( sdf <= 0 ) {
                    // surface!
                    if(subpix) {
                        lambda = lambda - delta_lambda * last_sdf / (sdf - last_sdf);
                    }
                    depth = lambda;
                    break;
                }
                lambda += delta_lambda; //fminf(trunc_dist, fmaxf(delta_lambda, sdf));
                last_sdf = sdf;
            }
        }

        // Compute normal
        const float3 pos_w = c_w + depth * ray_w;
        const float3 pos_v = (pos_w - boxmin) / (boxmax - boxmin);
        const float3 _n_w = vol.GetFractionalBackwardDiffDxDyDz(pos_v);
        const float len_n_w = length(_n_w);
        const float3 n_w = len_n_w > 0 ? _n_w / len_n_w : make_float3(0,0,1);
        const float3 n_c = mulSO3inv(T_wc,n_w);
        const float3 p_c = depth * ray_c;

        if(depth > 0 ) {
            const float lambient = 0.1;
            const float diffuse = 0.9;
            const float specular = 0.1;
            const float3 eyedir = -1.0f * p_c / length(p_c);
            const float3 _lightdir = make_float3(0.4,0.4,-1);
            const float3 lightdir = _lightdir / length(_lightdir);
            const float3 lightreflect = 2*dot(lightdir,n_c)*n_c + (-1.0) * lightdir;
            const float edotr = fmaxf(0,dot(eyedir,lightreflect));
            const float spec = edotr*edotr*edotr*edotr*edotr*edotr*edotr*edotr*edotr*edotr;

//          img(u,v) = (depth - near) / (far - near);
            imgdepth(u,v) = depth;
            img(u,v) = lambient + diffuse * dot(n_c, lightdir )  + specular * spec;
            norm(u,v) = make_float4(0.5,0.5,0.5,1) + make_float4(n_c, 0) /2.0f;
        }else{
            imgdepth(u,v) = 0.0f/0.0f;
            img(u,v) = 0;
            norm(u,v) = make_float4(0,0,0,1);
        }
    }
}

void Raycast(Image<float> depth, Image<float4> norm, Image<float> img, const Volume<SDF_t> vol, const float3 boxmin, const float3 boxmax, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float near, float far, float trunc_dist, bool subpix )
{    
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim, gridDim, img);
    KernRaycastSDF<<<gridDim,blockDim>>>(depth, norm, img, vol, boxmin, boxmax, T_wc, fu, fv, u0, v0, near, far, trunc_dist, subpix);
}

//////////////////////////////////////////////////////
// Raycast sphere
//////////////////////////////////////////////////////

__global__ void KernRaycastSphere(Image<float> depth, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float3 center, float r)
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    if( u < depth.w && v < depth.h ) {
        const float3 center_c = mulSE3inv(T_wc, center);
        const float3 ray_c = make_float3((u-u0)/fu,(v-v0)/fv, 1);

        const float ldotc = dot(ray_c,center_c);
        const float lsq = dot(ray_c,ray_c);
        const float csq = dot(center_c,center_c);
        float dm = (ldotc - sqrt(ldotc*ldotc - lsq*(csq - r*r) )) / lsq;
        depth(u,v) = dm;
    }
}

void RaycastSphere(Image<float> depth, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float3 center, float r)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim, gridDim, depth);
    KernRaycastSphere<<<gridDim,blockDim>>>(depth, T_wc, fu, fv, u0, v0, center, r);
}

}

#include "MatUtils.h"
#include "Image.h"
#include "Sdf.h"
#include "BoundedVolume.h"
#include "launch_utils.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Raycast SDF
//////////////////////////////////////////////////////

__global__ void KernRaycastSdf(Image<float> imgdepth, Image<float4> norm, Image<float> img, const BoundedVolume<SDF_t> vol, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float near, float far, bool subpix )
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    if( u < img.w && v < img.h ) {
        const float3 c_w = SE3Translation(T_wc);
        const float3 ray_c = make_float3((u-u0)/fu,(v-v0)/fv, 1);
        const float3 ray_w = mulSO3(T_wc, ray_c);

        // Raycast bounding box to find valid ray segment of sdf
        // http://www.cs.utah.edu/~awilliam/box/box.pdf
        const float3 tminbound = (vol.bbox.Min() - c_w) / ray_w;
        const float3 tmaxbound = (vol.bbox.Max() - c_w) / ray_w;
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
            float min_delta_lambda = vol.VoxelSizeUnits().x;
            float delta_lambda = 0;

            // March through space
            while(lambda < min_tmax) {
                const float3 pos_w = c_w + lambda * ray_w;
                const float sdf = vol.GetUnitsTrilinearClamped(pos_w);

                if( sdf <= 0 ) {
                    // surface!
                    if(subpix) {
                        lambda = lambda + delta_lambda * sdf / (last_sdf - sdf);
                    }
                    depth = lambda;
                    break;
                }
                delta_lambda = fmaxf(sdf, min_delta_lambda);
                lambda += delta_lambda;
                last_sdf = sdf;
            }
        }

        // Compute normal
        const float3 pos_w = c_w + depth * ray_w;
        const float3 _n_w = vol.GetUnitsBackwardDiffDxDyDz(pos_w);
        const float len_n_w = length(_n_w);
        const float3 n_w = len_n_w > 0 ? _n_w / len_n_w : make_float3(0,0,1);
        const float3 n_c = mulSO3inv(T_wc,n_w);
        const float3 p_c = depth * ray_c;

        if(depth > 0 ) {
            const float ambient = 0.4;
            const float diffuse = 0.4;
            const float specular = 0.2;
            const float3 eyedir = -1.0f * p_c / length(p_c);
            const float3 _lightdir = make_float3(0.4,0.4,-1);
            const float3 lightdir = _lightdir / length(_lightdir);
            const float ldotn = dot(lightdir,n_c);
            const float3 lightreflect = 2*ldotn*n_c + (-1.0) * lightdir;
            const float edotr = fmaxf(0,dot(eyedir,lightreflect));
            const float spec = edotr*edotr*edotr*edotr*edotr*edotr*edotr*edotr*edotr*edotr;

//          img(u,v) = (depth - near) / (far - near);
            imgdepth(u,v) = depth;
            img(u,v) = ambient + diffuse * ldotn  + specular * spec;
//            norm(u,v) = make_float4(0.5,0.5,0.5,1) + make_float4(n_c, 0) /2.0f;
            norm(u,v) = make_float4(-1.0f*n_c, 1);
        }else{
            imgdepth(u,v) = 0.0f/0.0f;
            img(u,v) = 0;
            norm(u,v) = make_float4(0,0,0,0);
        }
    }
}

void RaycastSdf(Image<float> depth, Image<float4> norm, Image<float> img, const BoundedVolume<SDF_t> vol, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float near, float far, bool subpix )
{    
    dim3 blockDim, gridDim;
//    InitDimFromOutputImageOver(blockDim, gridDim, img, 16, 16);
    InitDimFromOutputImageOver(blockDim, gridDim, img);
    KernRaycastSdf<<<gridDim,blockDim>>>(depth, norm, img, vol, T_wc, fu, fv, u0, v0, near, far, subpix);
    GpuCheckErrors();
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
    GpuCheckErrors();
}

//////////////////////////////////////////////////////
// Raycast box
//////////////////////////////////////////////////////

__global__ void KernRaycastBox(Image<float> depth, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, const BoundingBox bbox )
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    if( u < depth.w && v < depth.h ) {
        const float3 c_w = SE3Translation(T_wc);
        const float3 ray_c = make_float3((u-u0)/fu,(v-v0)/fv, 1);
        const float3 ray_w = mulSO3(T_wc, ray_c);

        // Raycast bounding box to find valid ray segment of sdf
        // http://www.cs.utah.edu/~awilliam/box/box.pdf
        const float3 tminbound = (bbox.Min() - c_w) / ray_w;
        const float3 tmaxbound = (bbox.Max() - c_w) / ray_w;
        const float3 tmin = fminf(tminbound,tmaxbound);
        const float3 tmax = fmaxf(tminbound,tmaxbound);
        const float max_tmin = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
        const float min_tmax = fminf(fminf(tmax.x, tmax.y), tmax.z);

        float d;

        // If ray intersects bounding box
        if(max_tmin < min_tmax ) {
            d = max_tmin;
        }else{
            d = 0.0f/0.0f;
        }

        depth(u,v) = d;
    }
}

void RaycastBox(Image<float> depth, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, const BoundingBox bbox )
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim, gridDim, depth);
    KernRaycastBox<<<gridDim,blockDim>>>(depth, T_wc, fu, fv, u0, v0, bbox);
    GpuCheckErrors();
}

}

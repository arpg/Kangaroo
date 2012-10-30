#include "Image.h"
#include "launch_utils.h"
#include "patch_score.h"
#include "ImageKeyframe.h"
#include "MatUtils.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Disparity to Depth Conversion
//////////////////////////////////////////////////////

__global__
void KernDisp2Depth(const Image<float> dIn, Image<float> dOut, float fu, float fBaseline, float fMinDisp)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    if( dOut.InBounds(x,y) ) {
        dOut(x,y) = dIn(x,y) >= fMinDisp ? fu * fBaseline / dIn(x,y) : 0.0f/0.0f;
    }
}

void Disp2Depth(Image<float> dIn, const Image<float> dOut, float fu, float fBaseline, float fMinDisp)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim, gridDim, dOut);
    KernDisp2Depth<<<gridDim,blockDim>>>( dIn, dOut, fu, fBaseline, fMinDisp );
}

template<typename Tout, typename Tin>
__global__ void KernFilterBadKinectData(Image<Tout> dFiltered, Image<Tin> dKinectDepth)
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const float z_mm = dKinectDepth(u,v);
    dFiltered(u,v) = z_mm >= 200 ? z_mm : NAN;
}

void FilterBadKinectData(Image<float> dFiltered, Image<unsigned short> dKinectDepth)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dFiltered);
    KernFilterBadKinectData<<<gridDim,blockDim>>>(dFiltered, dKinectDepth);
}

void FilterBadKinectData(Image<float> dFiltered, Image<float> dKinectDepth)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dFiltered);
    KernFilterBadKinectData<<<gridDim,blockDim>>>(dFiltered, dKinectDepth);
}

//////////////////////////////////////////////////////
// Kinect depthmap to vertex array
//////////////////////////////////////////////////////

template<typename Ti>
__global__ void KernDepthToVbo(
    Image<float4> dVbo, const Image<Ti> dDepth, float fu, float fv, float u0, float v0, float depthscale
) {
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const float kz = depthscale * dDepth(u,v);

    // (x,y,1) = kinv * (u,v,1)'
    const float z = kz;
    const float x = z * (u-u0) / fu;
    const float y = z * (v-v0) / fv;

    dVbo(u,v) = make_float4(x,y,z,1);
}

void DepthToVbo(Image<float4> dVbo, const Image<unsigned short> dDepth, float fu, float fv, float u0, float v0, float depthscale)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dVbo);
    KernDepthToVbo<unsigned short><<<gridDim,blockDim>>>(dVbo, dDepth, fu, fv, u0, v0, depthscale);
}

void DepthToVbo(Image<float4> dVbo, const Image<float> dDepth, float fu, float fv, float u0, float v0, float depthscale)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dVbo);
    KernDepthToVbo<float><<<gridDim,blockDim>>>(dVbo, dDepth, fu, fv, u0, v0, depthscale);
}

//////////////////////////////////////////////////////
// Create cbo for vbo based on projection into image
//////////////////////////////////////////////////////

__global__ void KernColourVbo(
    Image<uchar4> dId, const Image<float4> dPd, const Image<uchar3> dIc,
    Mat<float,3,4> KT_cd
) {
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    if(u < dId.w && v < dId.h )
    {
        const float4 Pd4 = dPd(u,v);

        const Mat<float,4,1> Pd = {Pd4.x, Pd4.y, Pd4.z, 1};
        const Mat<float,3,1> KPc = KT_cd * Pd;

        const Mat<float,2,1> pc = { KPc(0) / KPc(2), KPc(1) / KPc(2) };

        uchar4 Id;
        if( dIc.InBounds(pc(0), pc(1), 1) ) {
            const float3 v = dIc.GetBilinear<float3>(pc(0), pc(1));
            Id = make_uchar4(v.x, v.y, v.z, 255);
        }else{
            Id = make_uchar4(0,0,0,0);
        }
        dId(u,v) = Id;
    }
}

void ColourVbo(Image<uchar4> dId, const Image<float4> dPd, const Image<uchar3> dIc, const Mat<float,3,4> KT_cd )
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim, dId);
    KernColourVbo<<<gridDim,blockDim>>>(dId, dPd, dIc, KT_cd);
}


//////////////////////////////////////////////////////
// Create textured view given depth image and keyframes
//////////////////////////////////////////////////////

template<typename Tout, typename Tin>
__global__ void KernTextureDepth(Image<Tout> img, const ImageKeyframe<Tin> kf, const Image<float> depth, const Mat<float,3,4> T_wd, float fu, float fv, float u0, float v0)
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    if(u < img.w && v < img.h )
    {
        const float d = depth(u,v);
        const float3 P_w = T_wd * (d * make_float3((u-u0)/fu,(v-v0)/fv, 1));

        // project into kf
        const float3 _p_kf = kf.KiT_iw * P_w;
        const float2 p_kf = make_float2(_p_kf.x/_p_kf.z, _p_kf.y/_p_kf.z);
        if(kf.img.InBounds(p_kf,2)) {
            const float3 color = (1.0f/255.0f) * kf.img.template GetBilinear<float3>(p_kf);
            img(u,v) = make_float4(color,1);
        }else{
            img(u,v) = make_float4(0,0,0,1);
        }
    }
}

template<typename Tout, typename Tin>
void TextureDepth(Image<Tout> img, const ImageKeyframe<Tin> kf, const Image<float> depth, const Mat<float,3,4> T_wd, float fu, float fv, float u0, float v0)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim, img);
    KernTextureDepth<<<gridDim,blockDim>>>(img,kf,depth,T_wd,fu,fv,u0,v0);
}

template void TextureDepth<float4,uchar3>(Image<float4> img, const ImageKeyframe<uchar3> kf, const Image<float> depth, const Mat<float,3,4> T_wd, float fu, float fv, float u0, float v0);

}

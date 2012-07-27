#include "all.h"
#include "launch_utils.h"
#include "patch_score.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Kinect depthmap to vertex array
//////////////////////////////////////////////////////

template<typename Ti>
__global__ void KernKinectToVbo(
    Image<float4> dVbo, const Image<Ti> dKinectDepth, double fu, double fv, double u0, double v0
) {
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const float kz = dKinectDepth(u,v) / 1000.0f;

    // (x,y,1) = kinv * (u,v,1)'
    const float x = kz * (u-u0) / fu;
    const float y = kz * (v-v0) / fv;
    const float z = (kz > 0) ? kz : 0.0f/0.0f;

    dVbo(u,v) = make_float4(x,y,z,1);
}

void KinectToVbo(Image<float4> dVbo, const Image<unsigned short> dKinectDepth, double fu, double fv, double u0, double v0)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dVbo);
    KernKinectToVbo<unsigned short><<<gridDim,blockDim>>>(dVbo, dKinectDepth, fu, fv, u0, v0);
}

void KinectToVbo(Image<float4> dVbo, const Image<float> dKinectDepth, double fu, double fv, double u0, double v0)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dVbo);
    KernKinectToVbo<float><<<gridDim,blockDim>>>(dVbo, dKinectDepth, fu, fv, u0, v0);
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

    const float4 Pd4 = dPd(u,v);

    const Mat<float,4,1> Pd = {Pd4.x, Pd4.y, Pd4.z, 1};
    const Mat<float,3,1> KPc = KT_cd * Pd;

    const Mat<float,2,1> pc = { KPc(0) / KPc(2), KPc(1) / KPc(2) };

    uchar4 Id;
    if( dIc.InBounds(pc(0), pc(1), 1) ) {
        const float3 v = dIc.GetBilinear<float3>(pc(0), pc(1));
        Id = make_uchar4(v.z, v.y, v.x, 255);
    }else{
        Id = make_uchar4(0,0,0,0);
    }
    dId(u,v) = Id;
}

void ColourVbo(Image<uchar4> dId, const Image<float4> dPd, const Image<uchar3> dIc, const Mat<float,3,4> KT_cd )
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dId);
    KernColourVbo<<<gridDim,blockDim>>>(dId, dPd, dIc, KT_cd);
}

}

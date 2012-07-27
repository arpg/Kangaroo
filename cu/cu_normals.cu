#include "all.h"
#include "launch_utils.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Normals from VBO
//////////////////////////////////////////////////////

__global__ void KernNormalsFromVbo(Image<float4> dN, const Image<float4> dV)
{
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;

    const float4 Vc = dV(u,v);
    const float4 Vr = dV(u+1,v);
    const float4 Vu = dV(u,v+1);

    const float4 a = Vr - Vc;
    const float4 b = Vu - Vc;

    const float3 axb = make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );

    const float magaxb = length(axb);

    const float4 N = make_float4(axb.x/magaxb, axb.y/magaxb, axb.z/magaxb,1);

    dN(u,v) = N;
}

void NormalsFromVbo(Image<float4> dN, const Image<float4> dV)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dN);
    KernNormalsFromVbo<<<gridDim,blockDim>>>(dN, dV);
}

}

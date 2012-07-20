#pragma once

#include "all.h"
#include "launch_utils.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Plane Fitting
//////////////////////////////////////////////////////

__global__ void KernUpdateHeightmap(Image<float4> dHeightMap, const Image<float4> d3d, const Image<unsigned char> dImage,  const Mat<float,3,4> T_hc)
{
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    //calculate the position in heightmap coordinates
    float4 p_c = d3d(u,v);
    float3 p_h = make_float3(T_hc(0,0)*p_c.x + T_hc(0,1)*p_c.y + T_hc(0,2)*p_c.z + T_hc(0,3)*p_c.w,
                             T_hc(1,0)*p_c.x + T_hc(1,1)*p_c.y + T_hc(1,2)*p_c.z + T_hc(1,3)*p_c.w,
                             T_hc(2,0)*p_c.x + T_hc(2,1)*p_c.y + T_hc(2,2)*p_c.z + T_hc(2,3)*p_c.w);

    int x = (int)p_h.x;
    int y = (int)p_h.y;

    if(dHeightMap.InBounds(x,y) == true) {
        //calculate the variance of the measurement
        float v_z = p_c.z*0.01; //this is the perp. distance from the camera
        float4 oldVal = dHeightMap(x,y);
        float4 newVal = make_float4((oldVal.y * p_h.z + v_z * oldVal.x)/(oldVal.y+v_z),
                                    oldVal.y*v_z / (oldVal.y+v_z),
                                    (oldVal.y * dImage(u,v) + v_z * oldVal.z)/(oldVal.y+v_z),
                                    0.0);

        dHeightMap(x,y) = newVal;
    }
}

__global__ void KernColourHeightmap(Image<uchar4> dCbo, const Image<float4> dHeightMap)
{
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    float v_z = dHeightMap(u,v).z;
    dCbo(u,v) = make_uchar4(255,0,0,255);
}

__global__ void KernVboFromHeightmap(Image<float4> dVbo, const Image<float4> dHeightMap)
{
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    dVbo(u,v) = make_float4(u,v,0/*dHeightMap(u,v).z*/,1.0);
}


void InitHeightMap(Image<float4> dHeightMap)
{
    // initialize the heightmap
    dHeightMap.Fill(make_float4(0,1.0,0.0,0.0));
}

void VboFromHeightMap(Image<float4> dVbo, const Image<float4> dHeightMap)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dVbo);
    KernVboFromHeightmap<<<gridDim,blockDim>>>(dVbo,dHeightMap);
}

void ColourHeightMap(Image<uchar4> dCbo, const Image<float4> dHeightMap)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dCbo);
    KernColourHeightmap<<<gridDim,blockDim>>>(dCbo,dHeightMap);
}

void UpdateHeightMap(Image<float4> dHeightMap, const Image<float4> d3d, const Image<unsigned char> dImage, const Mat<float,3,4> T_hc)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, d3d);
    KernUpdateHeightmap<<<gridDim,blockDim>>>(dHeightMap,d3d,dImage,T_hc);
}
}

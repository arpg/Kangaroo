#pragma once

#include "Image.h"
#include "Mat.h"
#include "sampling.h"

namespace Gpu
{

//////////////////////////////////////////////////////
// Create Matlab Lookup table
//////////////////////////////////////////////////////

__global__ void KernCreateMatlabLookupTable(
    Image<float2> lookup, float fu, float fv, float u0, float v0, float k1, float k2
) {
    const uint u = blockIdx.x*blockDim.x + threadIdx.x;
    const uint v = blockIdx.y*blockDim.y + threadIdx.y;

    const float pnu = (u-u0) / fu;
    const float pnv = (v-v0) / fv;
    const float r = sqrt(pnu*pnu + pnv*pnv);
    const float rr = r*r;
    const float rf = 1 + k1*rr + k2*rr*rr; // + k3*rr*rr*rr;

    lookup(u,v) = make_float2(
        (pnu*rf /*+ 2*p1*pn.x*pn.y + p2*(rr + 2*pn.x*pn.x)*/) * fu + u0,
        (pnv*rf /*+ p1*(rr + 2*pn.y*pn.y) + 2*p2*pn.x*pn.y*/) * fv + v0
    );
}

void CreateMatlabLookupTable(
    Image<float2> lookup, float fu, float fv, float u0, float v0, float k1, float k2
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, lookup);
    KernCreateMatlabLookupTable<<<gridDim,blockDim>>>(lookup,fu,fv,u0,v0,k1,k2);
}

//////////////////////////////////////////////////////
// Create Matlab Lookup table applying homography
//////////////////////////////////////////////////////

__global__ void KernCreateMatlabLookupTable(
    Image<float2> lookup, float fu, float fv, float u0, float v0, float k1, float k2, Mat<float,9> H_on
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    // Apply homography H_on, moving New image to Original
    const float hdiv = H_on[6] * x + H_on[7] * y + H_on[8];
    const float u = (H_on[0] * x + H_on[1] * y + H_on[2]) / hdiv;
    const float v = (H_on[3] * x + H_on[4] * y + H_on[5]) / hdiv;

    // Apply distortion to achieve true image coordinates
    const float pnu = (u-u0) / fu;
    const float pnv = (v-v0) / fv;
    const float r = sqrt(pnu*pnu + pnv*pnv);
    const float rr = r*r;
    const float rf = 1 + k1*rr + k2*rr*rr; // + k3*rr*rr*rr;

    float2 pos = make_float2(
        (pnu*rf /*+ 2*p1*pn.x*pn.y + p2*(rr + 2*pn.x*pn.x)*/) * fu + u0,
        (pnv*rf /*+ p1*(rr + 2*pn.y*pn.y) + 2*p2*pn.x*pn.y*/) * fv + v0
    );

    // Clamp to image bounds
    pos.x = max(pos.x, 1.0f);
    pos.y = max(pos.y, 1.0f);
    pos.x = min(pos.x, lookup.w-2.0f);
    pos.y = min(pos.y, lookup.h-2.0f);

    lookup(x,y) = pos;
}

void CreateMatlabLookupTable(
        Image<float2> lookup, float fu, float fv, float u0, float v0, float k1, float k2, Mat<float,9> H_no
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, lookup);
    KernCreateMatlabLookupTable<<<gridDim,blockDim>>>(lookup,fu,fv,u0,v0,k1,k2,H_no);
}

//////////////////////////////////////////////////////
// Warp image using lookup table
//////////////////////////////////////////////////////

__global__ void KernWarp(
    Image<unsigned char> out, const Image<unsigned char> in, const Image<float2> lookup
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const float2 lu = lookup(x,y);
    out(x,y) = in.GetBilinear<float>(lu.x, lu.y);
}

void Warp(
    Image<unsigned char> out, const Image<unsigned char> in, const Image<float2> lookup
) {
    assert(out.w <= lookup.w && out.h <= lookup.h);
    assert(out.w <= in.w && out.h <= in.w);

    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, out);
    KernWarp<<<gridDim,blockDim>>>(out, in, lookup);

}

}

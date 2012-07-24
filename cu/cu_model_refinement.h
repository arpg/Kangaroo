#pragma once

#include "all.h"

#include "Mat.h"
#include "Image.h"

namespace Gpu {

__global__ void KernPoseRefinementFromDepthmap(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float4> dPr,
    const Mat<float,3,4> KT_lr, float c,
    Image<LeastSquaresSystem<float,6> > dSum, Image<float4> dDebug
){
    const unsigned int u = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int v = blockIdx.y*blockDim.y + threadIdx.y;

    const float4 Pr4 = dPr(u,v);
    // OpenGL point to what our intrinsics expects
    const Mat<float,4> Pr = {Pr4.x, -Pr4.y, -Pr4.z, 1};

    const Mat<float,3> KPl = KT_lr * Pr;
    const Mat<float,2> pl = {KPl(0)/KPl(2), KPl(1)/KPl(2)};
//    const Mat<float,2> pr = {u,v};

    float Il = dImgl.GetBilinear<float>(pl(0), pl(1));
    float Ir = dImgr(u,v);

    const float f = Il - Ir;

    const float debug = abs(f) / 255.0f;
    dDebug(u,v) = make_float4(debug,debug,debug,1);

    LeastSquaresSystem<float,6> sum;
    sum.SetZero();
    dSum(u,v) = sum;
}

LeastSquaresSystem<float,6> PoseRefinementFromDepthmap(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float4> dPr,
    const Mat<float,3,4> KT_lr, float c,
    Image<unsigned char> dWorkspace, Image<float4> dDebug
){
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dImgr);
    Image<LeastSquaresSystem<float,6> > dSum = dWorkspace.PackedImage<LeastSquaresSystem<float,6> >(dImgr.w, dImgr.h);

    KernPoseRefinementFromDepthmap<<<gridDim,blockDim>>>(dImgl, dImgr, dPr, KT_lr, c, dSum, dDebug );

    LeastSquaresSystem<float,6> sum;
    sum.SetZero();
    return thrust::reduce(dSum.begin(), dSum.end(), sum, thrust::plus<LeastSquaresSystem<float,6> >() );
}


}

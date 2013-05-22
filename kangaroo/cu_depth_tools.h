#pragma once

#include "Image.h"
#include "ImageKeyframe.h"

namespace roo
{

void Disp2Depth(Image<float> dIn, const Image<float> dOut, float fu, float fBaseline, float fMinDisp = 0.0);

void FilterBadKinectData(Image<float> dFiltered, Image<unsigned short> dKinectDepth);

void FilterBadKinectData(Image<float> dFiltered, Image<float> dKinectDepth);

void DepthToVbo( Image<float4> dVbo, const Image<unsigned short> dKinectDepth, ImageIntrinsics K, float scale = 1.0f);

void DepthToVbo( Image<float4> dVbo, const Image<float> dKinectDepth, ImageIntrinsics K, float scale = 1.0f);

inline void DepthToVbo( Image<float4> dVbo, const Image<unsigned short> dKinectDepth, float fu, float fv, float u0, float v0, float scale = 1.0f)
{
    DepthToVbo(dVbo, dKinectDepth, ImageIntrinsics(fu,fv,u0,v0), scale);
}

inline void DepthToVbo( Image<float4> dVbo, const Image<float> dKinectDepth, float fu, float fv, float u0, float v0, float scale = 1.0f)
{
    DepthToVbo(dVbo, dKinectDepth, ImageIntrinsics(fu,fv,u0,v0), scale);
}

void ColourVbo(Image<uchar4> dId, const Image<float4> dPd, const Image<uchar3> dIc, const Mat<float,3,4> KT_cd );

template<typename Tout, typename Tin>
void TextureDepth(Image<Tout> img, const ImageKeyframe<Tin> kf, const Image<float> depth, const Image<float4> norm, const Mat<float,3,4> T_wd, ImageIntrinsics Kdepth);

template<typename Tout, typename Tin, size_t N>
void TextureDepth(Image<Tout> img, const Mat<ImageKeyframe<Tin>,N> kfs, const Image<float> depth, const Image<float4> norm, const Image<float> phong, const Mat<float,3,4> T_wd, ImageIntrinsics Kdepth);

}

#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>
#include "ImageKeyframe.h"

namespace roo
{

KANGAROO_EXPORT
void Disp2Depth(Image<float> dIn, const Image<float> dOut, float fu, float fBaseline, float fMinDisp = 0.0);

KANGAROO_EXPORT
void FilterBadKinectData(Image<float> dFiltered, Image<unsigned short> dKinectDepth);

KANGAROO_EXPORT
void FilterBadKinectData(Image<float> dFiltered, Image<float> dKinectDepth);

template<typename T>
KANGAROO_EXPORT
void DepthToVbo( Image<float4> dVbo, const Image<T> dKinectDepth, ImageIntrinsics K, float scale = 1.0f);

template<typename T>
inline void DepthToVbo( Image<float4> dVbo, const Image<T> dKinectDepth, float fu, float fv, float u0, float v0, float scale = 1.0f)
{
    DepthToVbo(dVbo, dKinectDepth, ImageIntrinsics(fu,fv,u0,v0), scale);
}

KANGAROO_EXPORT
void ColourVbo(Image<uchar4> dId, const Image<float4> dPd, const Image<uchar3> dIc, const Mat<float,3,4> KT_cd );

template<typename Tout, typename Tin>
KANGAROO_EXPORT
void TextureDepth(Image<Tout> img, const ImageKeyframe<Tin> kf, const Image<float> depth, const Image<float4> norm, const Mat<float,3,4> T_wd, ImageIntrinsics Kdepth);

template<typename Tout, typename Tin, size_t N>
KANGAROO_EXPORT
void TextureDepth(Image<Tout> img, const Mat<ImageKeyframe<Tin>,N> kfs, const Image<float> depth, const Image<float4> norm, const Image<float> phong, const Mat<float,3,4> T_wd, ImageIntrinsics Kdepth);

}

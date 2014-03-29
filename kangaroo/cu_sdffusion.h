#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Mat.h>
#include <kangaroo/Image.h>
#include <kangaroo/BoundedVolume.h>
#include <kangaroo/ImageIntrinsics.h>
#include <kangaroo/Sdf.h>

namespace roo
{

KANGAROO_EXPORT
void SdfFuse(BoundedVolume<SDF_t> vol, Image<float> depth, Image<float4> norm, Mat<float,3,4> T_cw, ImageIntrinsics K, float trunc_dist, float maxw, float mincostheta );

KANGAROO_EXPORT
void SdfFuse(BoundedVolume<SDF_t> vol, BoundedVolume<float> colorVol, Image<float> depth, Image<float4> norm, Mat<float,3,4> T_cw, ImageIntrinsics K, Image<uchar3> img, Mat<float,3,4> T_iw, ImageIntrinsics Kimg,float trunc_dist, float max_w, float mincostheta);

KANGAROO_EXPORT
void SdfReset(BoundedVolume<SDF_t> vol, float trunc_dist);

KANGAROO_EXPORT
void SdfReset(BoundedVolume<float> vol);

KANGAROO_EXPORT
void SdfSphere(BoundedVolume<SDF_t> vol, float3 center, float r);

KANGAROO_EXPORT
void SdfDistance(Image<float> dist, Image<float> depth, BoundedVolume<SDF_t> vol, const Mat<float,3,4> T_wc, ImageIntrinsics K, float trunc_distance);

}

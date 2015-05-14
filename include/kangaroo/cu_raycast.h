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
void RaycastSdf(Image<float> depth, Image<float4> norm, Image<float> img, const BoundedVolume<SDF_t> vol, const Mat<float,3,4> T_wc, ImageIntrinsics K, float near, float far, float trunc_dist, bool subpix = true);

KANGAROO_EXPORT
void RaycastSdf(Image<float> depth, Image<float4> norm, Image<float> img, const BoundedVolume<SDF_t> vol, const BoundedVolume<float> colorVol, const Mat<float,3,4> T_wc, ImageIntrinsics K, float near, float far, float trunc_dist, bool subpix = true);

KANGAROO_EXPORT
void RaycastBox(Image<float> depth, const Mat<float,3,4> T_wc, ImageIntrinsics K, const BoundingBox bbox );

KANGAROO_EXPORT
void RaycastSphere(Image<float> depth, Image<float> img, const Mat<float,3,4> T_wc, ImageIntrinsics K, float3 center, float r);

KANGAROO_EXPORT
void RaycastPlane(Image<float> depth, Image<float> img, const Mat<float,3,4> T_wc, ImageIntrinsics K, const float3 n_w );

}

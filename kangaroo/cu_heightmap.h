#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Mat.h>
#include <kangaroo/Image.h>

namespace roo
{

KANGAROO_EXPORT
void VboFromHeightMap(Image<float4> dVbo, const Image<float4> dHeightMap);

KANGAROO_EXPORT
void VboWorldFromHeightMap(Image<float4> dVbo, const Image<float4> dHeightMap, const Mat<float,3,4> T_wh);

KANGAROO_EXPORT
void InitHeightMap(Image<float4> dHeightMap);

KANGAROO_EXPORT
void UpdateHeightMap(Image<float4> dHeightMap, const Image<float4> d3d, const Image<unsigned char> dImage, const Mat<float,3,4> T_hc, float min_height = -1E20, float max_height = 1E20, float max_distance = 1E20);

KANGAROO_EXPORT
void ColourHeightMap(Image<uchar4> dCbo, const Image<float4> dHeightMap);

KANGAROO_EXPORT
void GenerateWorldVboAndImageFromHeightmap(Image<float4> dVbo, Image<unsigned char> dImage, const Image<float4> dHeightMap, const Mat<float,3,4> T_wh);

}

#pragma once

// Hack to use gcc4.7 with cuda.
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cuda_runtime.h>

#include "Image.h"
#include "Pyramid.h"
#include "Volume.h"
#include "Mat.h"
#include "MatUtils.h"
#include "reduce.h"

namespace Gpu
{

template<typename To, typename Ti>
void ConvertImage(Image<To> dOut, const Image<Ti> dIn);

//////////////////////////////////////////////////////

void CreateMatlabLookupTable(Image<float2> lookup,
    float fu, float fv, float u0, float v0, float k1, float k2
);

void CreateMatlabLookupTable(Image<float2> lookup,
    float fu, float fv, float u0, float v0, float k1, float k2, Mat<float,9> H_no
);

//////////////////////////////////////////////////////

void Warp(
    Image<unsigned char> out, const Image<unsigned char> in, const Image<float2> lookup
);

//////////////////////////////////////////////////////

void DenseStereo(
    Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, int maxDisp, double acceptThresh
);

//////////////////////////////////////////////////////

void ReverseCheck(
    Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight
);

//////////////////////////////////////////////////////

void DenseStereoSubpixelRefine(Image<float> dDispOut, const Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight
);

//////////////////////////////////////////////////////

void DisparityImageCrossSection(
    Image<float4> dScore, Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, int y
);

//////////////////////////////////////////////////////

void FilterBadKinectData(Image<float> dFiltered, Image<unsigned short> dKinectDepth);
void DepthToVbo( Image<float4> dVbo, const Image<unsigned short> dKinectDepth, float fu, float fv, float u0, float v0, float scale = 1.0f);
void DepthToVbo( Image<float4> dVbo, const Image<float> dKinectDepth, float fu, float fv, float u0, float v0, float scale = 1.0f);

void DisparityImageToVbo(
    Image<float4> dVbo, const Image<float> dDisp, float baseline, float fu, float fv, float u0, float v0
);

void ColourVbo(Image<uchar4> dId, const Image<float4> dPd, const Image<uchar3> dIc, const Mat<float,3,4> KT_cd );

void NormalsFromVbo(Image<float4> dN, const Image<float4> dV);

//////////////////////////////////////////////////////

void GenerateTriangleStripIndexBuffer( Image<uint2> dIbo);

//////////////////////////////////////////////////////

LeastSquaresSystem<float,6> PoseRefinementFromVbo(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float4> dPr,
    const Mat<float,3,4> KT_lr, float c,
    Image<unsigned char> dWorkspace, Image<float4> dDebug
);

LeastSquaresSystem<float,6> PoseRefinementFromDisparity(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float> dDispr,
    const Mat<float,3,4> KT_lr, float c,
    float baseline, float fu, float fv, float u0, float v0,
    Image<unsigned char> dWorkspace, Image<float4> dDebug
);

LeastSquaresSystem<float,6> PoseRefinementProjectiveIcpPointPlane(
    const Image<float4> dPl,
    const Image<float4> dPr, const Image<float4> dNr,
    const Mat<float,3,4> KT_lr, const Mat<float,3,4> T_rl, float c,
    Image<unsigned char> dWorkspace, Image<float4> dDebug
);

LeastSquaresSystem<float,2*6> KinectCalibration(
    const Image<float4> dPl, const Image<uchar3> dIl,
    const Image<float4> dPr, const Image<uchar3> dIr,
    const Mat<float,3,4> KcT_cd, const Mat<float,3,4> T_lr,
    float c, Image<unsigned char> dWorkspace, Image<float4> dDebug
);

//////////////////////////////////////////////////////

LeastSquaresSystem<float,3> PlaneFitGN(const Image<float4> dVbo, Mat<float,3,3> Qinv, Mat<float,3> zhat, Image<unsigned char> dWorkspace, Image<float> dErr, float within, float c );

//////////////////////////////////////////////////////

void BilateralFilter(
    Image<float> dOut, const Image<float> dIn, float gs, float gr, uint size
);

void BilateralFilter(
    Image<float> dOut, const Image<unsigned char> dIn, float gs, float gr, uint size
);

void BilateralFilter(
    Image<float> dOut, const Image<unsigned short> dIn, float gs, float gr, uint size
);

void RobustBilateralFilter(
    Image<float> dOut, const Image<unsigned char> dIn, float gs, float gr, float go, uint size
);

//////////////////////////////////////////////////////

void MedianFilter3x3(
    Image<float> dOut, Image<float> dIn
);

void MedianFilter5x5(
    Image<float> dOut, Image<float> dIn
);

void MedianFilterRejectNegative5x5(
    Image<float> dOut, Image<float> dIn, int maxbad = 100
);

void MedianFilterRejectNegative7x7(
    Image<float> dOut, Image<float> dIn, int maxbad
);

void MedianFilterRejectNegative9x9(
    Image<float> dOut, Image<float> dIn, int maxbad
);

//////////////////////////////////////////////////////

void MakeAnaglyth(
    Image<uchar4> anaglyth,
    const Image<unsigned char> left, const Image<unsigned char> right,
    int shift = 0
);

//////////////////////////////////////////////////////

void VboFromHeightMap(Image<float4> dVbo, const Image<float4> dHeightMap);

void VboWorldFromHeightMap(Image<float4> dVbo, const Image<float4> dHeightMap, const Mat<float,3,4> T_wh);

void InitHeightMap(Image<float4> dHeightMap);

void UpdateHeightMap(Image<float4> dHeightMap, const Image<float4> d3d, const Image<unsigned char> dImage, const Mat<float,3,4> T_hc, float min_height = -1E20, float max_height = 1E20);

void ColourHeightMap(Image<uchar4> dCbo, const Image<float4> dHeightMap);

void GenerateWorldVboAndImageFromHeightmap(Image<float4> dVbo, Image<unsigned char> dImage, const Image<float4> dHeightMap, const Mat<float,3,4> T_wh);

//////////////////////////////////////////////////////

struct __align__(8) CostVolElem
{
    int n;
    float sum;
};

void InitCostVolume(Volume<CostVolElem> costvol );

void InitCostVolume(Volume<CostVolElem> dvol, Image<unsigned char> dimgl, Image<unsigned char> dimgr );

void AddToCostVolume(Volume<CostVolElem> vol, const Image<unsigned char> imgv,
    const Image<unsigned char> imgc, Mat<float,3,4> KT_cv,
    float fu, float fv, float u0, float v0,
    float minz, float maxz, int levels
);

void CostVolumeCrossSection(
    Image<float4> dScore, Volume<CostVolElem> dCostVol, int y
);

void FilterDispGrad(
    Image<float> dOut, Image<float> dIn, float threshold
);

//////////////////////////////////////////////////////

void Blur(Image<unsigned char> out, Image<unsigned char> in, Image<unsigned char> temp );

inline void Blur(Image<unsigned char> in_out, Image<unsigned char> temp )
{
    Blur(in_out,in_out,temp);
}

}

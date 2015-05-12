#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>
#include <kangaroo/Volume.h>
#include <kangaroo/CostVolElem.h>

namespace roo
{

//////////////////////////////////////////////////////

template<typename Tdisp, typename Tvol>
KANGAROO_EXPORT
void CostVolMinimum(Image<Tdisp> disp, Volume<Tvol> vol, unsigned maxDisp);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void DenseStereoTest(
    Image<float> dDisp, Image<unsigned char> dCamLeft, Image<unsigned char> dCamRight, int maxDisp
);

template<typename TDisp, typename TImg>
KANGAROO_EXPORT
void DenseStereo(
    Image<TDisp> dDisp, const Image<TImg> dCamLeft, const Image<TImg> dCamRight, TDisp maxDisp, float acceptThresh, int score_rad
);

KANGAROO_EXPORT
void DenseStereoSubpix(
    Image<float> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, float maxDisp, float dispStep, float acceptThresh, int score_rad, bool score_normed
);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void LeftRightCheck(Image<char> dispL, Image<char> dispR, int sd = -1, int maxDiff = 0);

KANGAROO_EXPORT
void LeftRightCheck(Image<float> dispL, Image<float> dispR, float sd = -1, float maxDiff = 0.5);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void DenseStereoSubpixelRefine(Image<float> dDispOut, const Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight
);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void DisparityImageCrossSection(
    Image<float4> dScore, Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, int y
);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void DisparityImageToVbo(
    Image<float4> dVbo, const Image<float> dDisp, float baseline, float fu, float fv, float u0, float v0
);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void CostVolumeFromStereoTruncatedAbsAndGrad(Volume<float> dvol, Image<float> dimgl, Image<float> dimgr, float sd, float alpha, float r1, float r2 );

KANGAROO_EXPORT
void CostVolumeZero(Volume<CostVolElem> costvol );

KANGAROO_EXPORT
void CostVolumeFromStereo(Volume<CostVolElem> dvol, Image<unsigned char> dimgl, Image<unsigned char> dimgr );

KANGAROO_EXPORT
void CostVolumeAdd(Volume<CostVolElem> vol, const Image<unsigned char> imgv,
    const Image<unsigned char> imgc, Mat<float,3,4> KT_cv,
    float fu, float fv, float u0, float v0,
    float baseline, int levels
);

KANGAROO_EXPORT
void CostVolMinimum(Image<float> disp, Volume<CostVolElem> vol);

KANGAROO_EXPORT
void CostVolMinimumSubpix(Image<float> disp, Volume<float> vol, unsigned maxDisp, float sd);

KANGAROO_EXPORT
void CostVolMinimumSquarePenaltySubpix(Image<float> imga, Volume<float> vol, Image<float> imgd, unsigned maxDisp, float sd, float lambda, float theta);

KANGAROO_EXPORT
void ExponentialEdgeWeight(Image<float> imgw, const Image<float> imgi, float alpha, float beta);

KANGAROO_EXPORT
void CostVolumeCrossSection(
    Image<float> dScore, Volume<CostVolElem> dCostVol, int y
);

//////////////////////////////////////////////////////

KANGAROO_EXPORT
void FilterDispGrad(
    Image<float> dOut, Image<float> dIn, float threshold
);

}

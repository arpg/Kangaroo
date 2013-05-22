#pragma once

#include "Image.h"
#include "Volume.h"
#include "CostVolElem.h"

namespace roo
{

//////////////////////////////////////////////////////

template<typename Tdisp, typename Tvol>
void CostVolMinimum(Image<Tdisp> disp, Volume<Tvol> vol, unsigned maxDisp);

//////////////////////////////////////////////////////

void DenseStereoTest(
    Image<float> dDisp, Image<unsigned char> dCamLeft, Image<unsigned char> dCamRight, int maxDisp
);

template<typename TDisp, typename TImg>
void DenseStereo(
    Image<TDisp> dDisp, const Image<TImg> dCamLeft, const Image<TImg> dCamRight, TDisp maxDisp, float acceptThresh, int score_rad
);

void DenseStereoSubpix(
    Image<float> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, float maxDisp, float dispStep, float acceptThresh, int score_rad, bool score_normed
);

//////////////////////////////////////////////////////

void LeftRightCheck(Image<char> dispL, Image<char> dispR, int sd = -1, int maxDiff = 0);

void LeftRightCheck(Image<float> dispL, Image<float> dispR, float sd = -1, float maxDiff = 0.5);

//////////////////////////////////////////////////////

void DenseStereoSubpixelRefine(Image<float> dDispOut, const Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight
);

//////////////////////////////////////////////////////

void DisparityImageCrossSection(
    Image<float4> dScore, Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, int y
);

//////////////////////////////////////////////////////

void DisparityImageToVbo(
    Image<float4> dVbo, const Image<float> dDisp, float baseline, float fu, float fv, float u0, float v0
);

//////////////////////////////////////////////////////

void CostVolumeFromStereoTruncatedAbsAndGrad(Volume<float> dvol, Image<float> dimgl, Image<float> dimgr, float sd, float alpha, float r1, float r2 );

void CostVolumeZero(Volume<CostVolElem> costvol );

void CostVolumeFromStereo(Volume<CostVolElem> dvol, Image<unsigned char> dimgl, Image<unsigned char> dimgr );

void CostVolumeAdd(Volume<CostVolElem> vol, const Image<unsigned char> imgv,
    const Image<unsigned char> imgc, Mat<float,3,4> KT_cv,
    float fu, float fv, float u0, float v0,
    float baseline, int levels
);

void CostVolMinimum(Image<float> disp, Volume<CostVolElem> vol);

void CostVolMinimumSubpix(Image<float> disp, Volume<float> vol, unsigned maxDisp, float sd);

void CostVolMinimumSquarePenaltySubpix(Image<float> imga, Volume<float> vol, Image<float> imgd, unsigned maxDisp, float sd, float lambda, float theta);

void ExponentialEdgeWeight(Image<float> imgw, const Image<float> imgi, float alpha, float beta);

void CostVolumeCrossSection(
    Image<float> dScore, Volume<CostVolElem> dCostVol, int y
);

//////////////////////////////////////////////////////

void FilterDispGrad(
    Image<float> dOut, Image<float> dIn, float threshold
);

}

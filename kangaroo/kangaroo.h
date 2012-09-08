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
#include "CudaTimer.h"

namespace Gpu
{

//////////////////////////////////////////////////////

template<typename To, typename Ti>
void ConvertImage(Image<To> dOut, const Image<Ti> dIn);

//////////////////////////////////////////////////////

template<typename Tout, typename Tin1, typename Tin2>
void Add(Image<Tout> out, Image<Tin1> in1, Image<Tin2> in2);

template<typename Tout, typename Tin1, typename Tin2, typename Tup>
void SubtractAdd(Image<Tout> out, Image<Tin1> in1, Image<Tin2> in2, Tup offset );

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

void Census(Image<unsigned long> census, Image<unsigned char> img);

void Census(Image<ulong2> census, Image<unsigned char> img);

void Census(Image<ulong4> census, Image<unsigned char> img);

//////////////////////////////////////////////////////

void CensusStereo(Image<char> disp, Image<unsigned long> left, Image<unsigned long> right, int maxDisp);

void CensusStereoVolume(Volume<unsigned short> vol, Image<unsigned long> left, Image<unsigned long> right, int maxDisp);

void CensusStereoVolume(Volume<unsigned short> vol, Image<ulong2> left, Image<ulong2> right, int maxDisp);

void CensusStereoVolume(Volume<unsigned short> vol, Image<ulong4> left, Image<ulong4> right, int maxDisp);

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

void LeftRightCheck(Image<char> dispL, Image<char> dispR, int maxDiff = 0);

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

void SumSpeedTest(
    Image<unsigned char> dWorkspace, int w, int h, int blockx, int blocky
);

//////////////////////////////////////////////////////

LeastSquaresSystem<float,3> PlaneFitGN(const Image<float4> dVbo, Mat<float,3,3> Qinv, Mat<float,3> zhat, Image<unsigned char> dWorkspace, Image<float> dErr, float within, float c );

//////////////////////////////////////////////////////

template<typename To, typename Ti>
void BilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, uint size
);

template<typename To, typename Ti, typename Ti2>
void BilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, const Image<Ti2> dImg, float gs, float gr, float gc, uint size
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

void UpdateHeightMap(Image<float4> dHeightMap, const Image<float4> d3d, const Image<unsigned char> dImage, const Mat<float,3,4> T_hc, float min_height = -1E20, float max_height = 1E20, float max_distance = 1E20);

void ColourHeightMap(Image<uchar4> dCbo, const Image<float4> dHeightMap);

void GenerateWorldVboAndImageFromHeightmap(Image<float4> dVbo, Image<unsigned char> dImage, const Image<float4> dHeightMap, const Mat<float,3,4> T_wh);

//////////////////////////////////////////////////////

struct __align__(8) CostVolElem
{
    inline __host__ __device__
    operator float() {
        return n > 0 ? sum / n : 1E30;
    }

    int n;
    float sum;
};

void CostVolumeZero(Volume<CostVolElem> costvol );

void CostVolumeFromStereo(Volume<CostVolElem> dvol, Image<unsigned char> dimgl, Image<unsigned char> dimgr );

void CostVolumeAdd(Volume<CostVolElem> vol, const Image<unsigned char> imgv,
    const Image<unsigned char> imgc, Mat<float,3,4> KT_cv,
    float fu, float fv, float u0, float v0,
    float baseline, int levels
);

void CostVolMinimum(Image<float> disp, Volume<CostVolElem> vol);

void CostVolumeCrossSection(
    Image<float4> dScore, Volume<CostVolElem> dCostVol, int y
);

//////////////////////////////////////////////////////

void SemiGlobalMatching(Volume<float> volH, Volume<unsigned short> volC, Image<unsigned char> left, int maxDisp, float P1, float P2, bool dohoriz, bool dovert, bool doreverse);

void SemiGlobalMatching(Volume<float> volH, Volume<CostVolElem> volC, Image<unsigned char> left, int maxDisp, float P1, float P2, bool dohoriz, bool dovert, bool doreverse);

//////////////////////////////////////////////////////

void FilterDispGrad(
    Image<float> dOut, Image<float> dIn, float threshold
);

//////////////////////////////////////////////////////

void Blur(Image<unsigned char> out, Image<unsigned char> in, Image<unsigned char> temp );

inline void Blur(Image<unsigned char> in_out, Image<unsigned char> temp )
{
    Blur(in_out,in_out,temp);
}

template<typename Tout, typename Tin, unsigned MAXRAD, unsigned MAXIMGDIM>
void GaussianBlur(Image<Tout> out, Image<Tin> in, Image<Tout> temp, float sigma);

//////////////////////////////////////////////////////

LeastSquaresSystem<float,3> ManhattenLineCost(
    Image<float4> out, Image<float4> out2, const Image<unsigned char> in,
    Mat<float,3,3> Rhat, float fu, float fv, float u0, float v0,
    float cut, float scale, float min_grad,
    Image<unsigned char> dWorkspace
);

//////////////////////////////////////////////////////

template<typename Tout, typename Tin>
void Transpose(Image<Tout> out, Image<Tin> in);

template<typename Tout, typename Tin>
void PrefixSumRows(Image<Tout> out, Image<Tin> in);

template<typename Tout, typename Tin>
void BoxFilterIntegralImage(Image<Tout> out, Image<Tin> IntegralImageT, int rad);

}

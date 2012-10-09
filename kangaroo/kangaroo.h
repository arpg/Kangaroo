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

void Disp2Depth(Image<float> dIn, const Image<float> dOut, float fu, float fBaseline, float fMinDisp = 0.0);

//////////////////////////////////////////////////////

template<typename T>
void Fill(Image<T> img, T val);

template<typename Tout, typename Tin, typename Tup>
void ElementwiseScaleBias(Image<Tout> b, const Image<Tin> a, float s, Tup offset=0);

template<typename Tout, typename Tin1, typename Tin2, typename Tup>
void ElementwiseAdd(Image<Tout> c, Image<Tin1> a, Image<Tin2> b, Tup sa=1, Tup sb=1, Tup offset=0 );

template<typename Tout, typename Tin1, typename Tin2, typename Tup>
void ElementwiseMultiply(Image<Tout> c, Image<Tin1> a, Image<Tin2> b, Tup scalar=1, Tup offset=0 );

template<typename Tout, typename Tin1, typename Tin2, typename Tup>
void ElementwiseDivision(Image<Tout> c, const Image<Tin1> a, const Image<Tin2> b, Tup sa=0, Tup sb=0, Tup scalar=1, Tup offset=0);

template<typename Tout, typename Tin, typename Tup>
void ElementwiseSquare(Image<Tout> b, const Image<Tin> a, Tup scalar=1, Tup offset=0 );

template<typename Tout, typename Tin1, typename Tin2, typename Tin3, typename Tup>
void ElementwiseMultiplyAdd(Image<Tout> d, const Image<Tin1> a, const Image<Tin2> b, const Image<Tin3> c, Tup sab=1, Tup sc=1, Tup offset=0);

//////////////////////////////////////////////////////

template<typename Tout, typename T>
Tout ImageL1(Image<T> img, Image<unsigned char> scratch);

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

void Census(Image<unsigned long> census, Image<float> img);
void Census(Image<ulong2> census, Image<float> img);
void Census(Image<ulong4> census, Image<float> img);

//////////////////////////////////////////////////////

void CensusStereo(Image<char> disp, Image<unsigned long> left, Image<unsigned long> right, int maxDisp);

template<typename Tvol, typename T>
void CensusStereoVolume(Volume<Tvol> vol, Image<T> left, Image<T> right, int maxDisp, float sd);

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

void FilterBadKinectData(Image<float> dFiltered, Image<unsigned short> dKinectDepth);
void FilterBadKinectData(Image<float> dFiltered, Image<float> dKinectDepth);
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

LeastSquaresSystem<float,6> PoseRefinementFromDisparityESM(
        const Image<unsigned char> dImgl,
        const Image<unsigned char> dImgr, const Image<float> dDispr,
        const Mat<float,3,4> KT_lr, float c,
        float baseline, float fu, float fv, float u0, float v0,
        Image<unsigned char> dWorkspace, Image<float4> dDebug,
        const bool bDiscardMaxMin = false
        );

LeastSquaresSystem<float,6> PoseRefinementFromDepthESM(
    const Image<unsigned char> dImgl,
    const Image<unsigned char> dImgr, const Image<float> dDispr,
    const Mat<float,3,4> KT_lr, float c,
    float fu, float fv, float u0, float v0,
    Image<unsigned char> dWorkspace, Image<float4> dDebug,
    const bool bDiscardMaxMin = false
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

template<typename To, typename Ti>
void BilateralFilter(
    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, uint size, Ti minval
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

void CostVolumeFromStereoTruncatedAbsAndGrad(Volume<float> dvol, Image<float> dimgl, Image<float> dimgr, float sd, float alpha, float r1, float r2 );

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

void CostVolMinimumSubpix(Image<float> disp, Volume<float> vol, unsigned maxDisp, float sd);

void CostVolMinimumSquarePenaltySubpix(Image<float> imga, Volume<float> vol, Image<float> imgd, unsigned maxDisp, float sd, float lambda, float theta);

void ExponentialEdgeWeight(Image<float> imgw, const Image<float> imgi, float alpha, float beta);

void CostVolumeCrossSection(
    Image<float> dScore, Volume<CostVolElem> dCostVol, int y
);

//////////////////////////////////////////////////////

template<typename TH, typename TC, typename Timg>
void SemiGlobalMatching(Volume<TH> volH, Volume<TC> volC, Image<Timg> left, int maxDisp, float P1, float P2, bool dohoriz, bool dovert, bool doreverse);

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

template<typename OT, typename IT, typename KT, typename ACC>
void Convolution(
    Image<OT> out,  Image<IT> in,  Image<KT> kern, int kx, int ky
);

//////////////////////////////////////////////////////

template<typename Tout, typename Tin>
void Transpose(Image<Tout> out, Image<Tin> in);

template<typename Tout, typename Tin>
void PrefixSumRows(Image<Tout> out, Image<Tin> in);

//////////////////////////////////////////////////////

template<typename Tout, typename Tin>
void BoxFilterIntegralImage(Image<Tout> out, Image<Tin> IntegralImageT, int rad);

template<typename Tout, typename Tin, typename TSum>
void BoxFilter(Image<Tout> out, Image<Tin> in, Image<unsigned char> scratch, int rad)
{
    Image<TSum> RowPrefixSum = scratch.AlignedImage<TSum>(in.w, in.h);
    PrefixSumRows<TSum,Tin>(RowPrefixSum, in);

    Image<TSum> RowPrefixSumT = out.template AlignedImage<TSum>(in.h, in.w);
    Transpose<TSum,TSum>(RowPrefixSumT,RowPrefixSum);

    Image<TSum> IntegralImageT = scratch.template AlignedImage<TSum>(in.h, in.w);
    PrefixSumRows<TSum,TSum>(IntegralImageT, RowPrefixSumT);

    BoxFilterIntegralImage<Tout,TSum>(out,IntegralImageT,rad);
}

//////////////////////////////////////////////////////

template<typename Tout, typename Tin, typename TSum>
void ComputeMeanVarience(Image<Tout> varI, Image<Tout> meanII, Image<Tout> meanI, const Image<Tin> I, Image<unsigned char> Scratch, int rad)
{
    // mean_I = boxfilter(I, r) ./ N;
    BoxFilter<float,float,float>(meanI,I,Scratch,rad);

    // mean_II = boxfilter(I.*I, r) ./ N;
    Image<Tout>& II = varI; // temp
    ElementwiseSquare<float,float,float>(II,I);
    BoxFilter<float,float,float>(meanII,II,Scratch,rad);

    // var_I = mean_II - mean_I .* mean_I;
    ElementwiseMultiplyAdd<float,float,float,float,float>(varI, meanI, meanI, meanII,-1);
}

inline void ComputeCovariance(Image<float> covIP, Image<float> meanIP, Image<float> meanP, const Image<float> P, const Image<float> meanI, const Image<float> I, Image<unsigned char> Scratch, int rad)
{
    // mean_p = boxfilter(p, r) ./ N;
    BoxFilter<float,float,float>(meanP,P,Scratch,rad);

    // mean_Ip = boxfilter(I.*p, r) ./ N;
    Image<float>& IP = covIP; // temp
    ElementwiseMultiply<float,float,float,float>(IP,I,P);
    BoxFilter<float,float,float>(meanIP,IP,Scratch,rad);

    // cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
    ElementwiseMultiplyAdd<float,float,float,float,float>(covIP, meanI, meanP, meanIP, -1);
}

//////////////////////////////////////////////////////

inline void GuidedFilter(Image<float> q, const Image<float> covIP, const Image<float> varI, const Image<float> meanP, const Image<float> meanI, const Image<float> I, Image<unsigned char> Scratch, Image<float> tmp1, Image<float> tmp2, Image<float> tmp3, int rad, float eps)
{
    Image<float>& a = tmp1;
    Image<float>& b = tmp2;
    Image<float>& meana = tmp3;
    Image<float>& meanb = tmp1;

    // a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
    ElementwiseDivision<float,float,float,float>(a, covIP, varI, 0, eps);

    // mean_a = boxfilter(a, r) ./ N;
    BoxFilter<float,float,float>(meana,a,Scratch,rad);

    // b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
    ElementwiseMultiplyAdd<float,float,float,float,float>(b, a, meanI, meanP, -1);

    // mean_b = boxfilter(b, r) ./ N;
    BoxFilter<float,float,float>(meanb,b,Scratch,rad);

    // q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
    ElementwiseMultiplyAdd<float,float,float,float,float>(q,meana,I,meanb);
}

//////////////////////////////////////////////////////

void SegmentTest(
    Image<unsigned char> out, const Image<unsigned char> img, unsigned char threshold, unsigned char min_segment_len
);

void HarrisScore(
    Image<float> out, const Image<unsigned char> img, float lambda = 0.04
);

void NonMaximalSuppression(Image<unsigned char> out, Image<float> scores, int rad, float threshold);

//////////////////////////////////////////////////////

template<typename T>
void PaintCircle(Image<T> img, T val, float x, float y, float r );

//////////////////////////////////////////////////////

struct SDF_t {
    inline __host__ __device__ SDF_t() {}
    inline __host__ __device__ SDF_t(float v) : val(v), n(1) {}
    inline __host__ __device__ SDF_t(float v, int n) : val(v), n(n) {}
    inline __host__ __device__ operator float() const {
        return val / n;
    }
    float val;
    int n;
};

void Raycast(Image<float> img, const Volume<SDF_t> vol, const float3 boxmin, const float3 boxmax, const Mat<float,3,4> T_wc, float fu, float fv, float u0, float v0, float near, float far, bool subpix );

void SDFSphere(Volume<SDF_t> vol, float3 vol_min, float3 vol_max, float3 center, float r);

}

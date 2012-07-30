#include "all.h"
#include "launch_utils.h"
#include "patch_score.h"

namespace Gpu
{

const int DefaultRad = 2;
typedef SSNDPatchScore<float,DefaultRad,ImgAccessClamped> DefaultSafeScoreType;
//typedef SinglePixelSqPatchScore<float,ImgAccessRaw> DefaultSafeScoreType;

//////////////////////////////////////////////////////
// Scanline rectified dense stereo
//////////////////////////////////////////////////////

template<typename TD, typename TI, typename Score>
__global__ void KernDenseStereo(
    Image<TD> dDisp, Image<TI> dCamLeft, Image<TI> dCamRight, int maxDisp, double acceptThresh
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    // Search for best matching pixel
    int bestDisp = 0;
    float bestScore = 1E+36;
    float sndBestScore = 1E+37;

    maxDisp = min(maxDisp, x);

    for(int c = 0; c <= maxDisp; ++c ) {
        const int rx = x-c;
        const float score =  Score::Score(dCamLeft, x,y, dCamRight, rx, y);
        if(score < bestScore) {
            sndBestScore = bestScore;
            bestScore = score;
            bestDisp = c;
        }else if( score < sndBestScore) {
            sndBestScore = score;
        }
    }

    const bool valid = (bestScore * acceptThresh) < sndBestScore;

    dDisp(x,y) = valid ? bestDisp : 0;
}

void DenseStereo(
    Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, int maxDisp, double acceptThresh
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dDisp);
    KernDenseStereo<unsigned char, unsigned char, DefaultSafeScoreType><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight,maxDisp,acceptThresh);
}

//////////////////////////////////////////////////////
// Visualise cross section of disparity image
//////////////////////////////////////////////////////

template<typename TD, typename TI, typename Score>
__global__ void KernDisparityImageCrossSection(
    Image<TD> dScore, Image<unsigned char> dDisp, Image<TI> dCamLeft, Image<TI> dCamRight, int y
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint c = blockIdx.y*blockDim.y + threadIdx.y;

    const int rx = x-c;
    const float score = ( 0<= rx && rx < dCamRight.w ) ? Score::Score(dCamLeft, x,y, dCamRight, rx, y) : 0;

    const unsigned char mindisp = dDisp(x,y);
    const float show = sqrt(score / Score::area) / 255.0f;

    dScore(x,c) = show * make_float4( 1,1,1,1);
}

void DisparityImageCrossSection(
    Image<float4> dScore, Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, int y
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dScore);
    KernDisparityImageCrossSection<float4, unsigned char, DefaultSafeScoreType><<<gridDim,blockDim>>>(dScore, dDisp, dCamLeft, dCamRight, y);
}

//////////////////////////////////////////////////////
// Scanline rectified dense stereo sub-pixel refinement
//////////////////////////////////////////////////////

template<typename TDo, typename TDi, typename TI, typename Score>
__global__ void KernDenseStereoSubpixelRefine(
    Image<TDo> dDispOut, const Image<TDi> dDisp, const Image<TI> dCamLeft, const Image<TI> dCamRight
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const int bestDisp = dDisp(x,y);

    // Ignore things at infinity (and outliers marked with 0)
    if(bestDisp <1) {
        dDispOut(x,y) = -1;
        return;
    }

    // Fit parabola to neighbours
    const float d1 = bestDisp+1;
    const float d2 = bestDisp;
    const float d3 = bestDisp-1;
    const float s1 = Score::Score(dCamLeft, x,y, dCamRight, x-d1,y);
    const float s2 = Score::Score(dCamLeft, x,y, dCamRight, x-d2,y);
    const float s3 = Score::Score(dCamLeft, x,y, dCamRight, x-d3,y);

    // Cooefficients of parabola through (d1,s1),(d2,s2),(d3,s3)
    const float denom = (d1 - d2)*(d1 - d3)*(d2 - d3);
    const float A = (d3 * (s2 - s1) + d2 * (s1 - s3) + d1 * (s3 - s2)) / denom;
    const float B = (d3*d3 * (s1 - s2) + d2*d2 * (s3 - s1) + d1*d1 * (s2 - s3)) / denom;
//    const float C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

    // Minima of parabola
    const float newDisp = -B / (2*A);

    // Check that minima is sensible. Otherwise assume bad data.
    if( d3 < newDisp && newDisp < d1 ) {
        dDispOut(x,y) = newDisp;
    }else{
//        dDisp(x,y) = bestDisp / maxDisp;
        dDispOut(x,y) = -1;
    }
}

void DenseStereoSubpixelRefine(
    Image<float> dDispOut, const Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dDisp);
    KernDenseStereoSubpixelRefine<float,unsigned char,unsigned char, DefaultSafeScoreType><<<gridDim,blockDim>>>(dDispOut, dDisp, dCamLeft, dCamRight);
}

//////////////////////////////////////////////////////
// Upgrade disparity image to vertex array
//////////////////////////////////////////////////////

__global__ void KernDisparityImageToVbo(
    Image<float4> dVbo, const Image<float> dDisp, float baseline, float fu, float fv, float u0, float v0
) {
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const float invalid = 0.0f/0.0f;

    const float disp = dDisp(u,v);
    const float z = disp > 2 ? fu * baseline / disp : invalid;

    // (x,y,1) = kinv * (u,v,1)'
    const float x = z * (u-u0) / fu;
    const float y = z * (v-v0) / fv;

    dVbo(u,v) = make_float4(x,y,z,1);
}

void DisparityImageToVbo(Image<float4> dVbo, const Image<float> dDisp, float baseline, float fu, float fv, float u0, float v0)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dVbo);
    KernDisparityImageToVbo<<<gridDim,blockDim>>>(dVbo, dDisp, baseline, fu, fv, u0, v0);
}

//////////////////////////////////////////////////////
// Cost Volume
//////////////////////////////////////////////////////

void InitCostVolume(Volume<CostVolElem> costvol )
{
    CostVolElem initial;
    initial.sum = 0;
    initial.n = 0;
    costvol.Fill(initial);
}

//////////////////////////////////////////////////////

template<typename TD, typename TI, typename Score>
__global__ void KernInitCostVolumeFromStereo(
    Volume<CostVolElem> dvol, Image<TI> dimgl, Image<TI> dimgr
) {
    const uint u = blockIdx.x*blockDim.x + threadIdx.x;
    const uint v = blockIdx.y*blockDim.y + threadIdx.y;
    const uint d = blockIdx.z*blockDim.z + threadIdx.z;

    CostVolElem elem;
    elem.sum = Score::Score(dimgl, u,v, dimgr, u-d, v) / Score::area;
    elem.n = 1;

    dvol(u,v,d) = elem;
}

void InitCostVolume(Volume<CostVolElem> dvol, Image<unsigned char> dimgl, Image<unsigned char> dimgr )
{
    dim3 blockDim(8,8,8);
    dim3 gridDim(dvol.w / blockDim.x, dvol.h / blockDim.y, dvol.d / blockDim.z);
    KernInitCostVolumeFromStereo<unsigned char, unsigned char, DefaultSafeScoreType><<<gridDim,blockDim>>>(dvol,dimgl,dimgr);
}

//////////////////////////////////////////////////////

template<typename TI, typename Score>
__global__ void KernAddToCostVolume(
    Volume<CostVolElem> dvol, const Image<TI> dimgv,
    const Image<TI> dimgc, Mat<float,3,4> KT_cv,
    float fu, float fv, float u0, float v0,
    float minz, float maxz, int /*levels*/
){
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const int d = blockIdx.z*blockDim.z + threadIdx.z;

    float3 Pv;
    Pv.z = fu / (minz + maxz*d);
    Pv.x = Pv.z * (u-u0) / fu;
    Pv.y = Pv.z * (v-v0) / fv;

    const float2 pc = dn(KT_cv * Pv);

    if( dimgc.InBounds(pc.x, pc.y,5) ) {
//        vol(u,v,d) = 1.0f;
//        const float score =  Score::Score(imgv, u,v, imgc, pc.x, pc.y) / (float)(Score::area);
        const float score = (dimgv(u,v) - dimgc.template GetBilinear<float>(pc)) / 255.0f;
        CostVolElem elem = dvol(u,v,d);
        elem.sum += score;
        elem.n += 1;
        dvol(u,v,d) = elem;
    }
}

void AddToCostVolume(Volume<CostVolElem> dvol, const Image<unsigned char> dimgv,
    const Image<unsigned char> dimgc, Mat<float,3,4> KT_cv,
    float fu, float fv, float u0, float v0,
    float minz, float maxz, int levels
) {
    dim3 blockDim(8,8,8);
    dim3 gridDim(dvol.w / blockDim.x, dvol.h / blockDim.y, dvol.d / blockDim.z);
    KernAddToCostVolume<unsigned char, SSNDPatchScore<float,DefaultRad,ImgAccessRaw> ><<<gridDim,blockDim>>>(dvol,dimgv,dimgc, KT_cv, fu,fv,u0,v0, minz,maxz, levels);
}

//////////////////////////////////////////////////////

__global__ void KernCostVolumeCrossSection(
    Image<float4> dScore, Image<CostVolElem> dCostVolSlice
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint d = blockIdx.y*blockDim.y + threadIdx.y;

    if( dCostVolSlice.InBounds(x,d) )
    {
        CostVolElem elem = dCostVolSlice(x,d);
        const float score = (elem.sum / elem.n) / 255.0f;
        dScore(x,d) = make_float4(score,score,score,1);
    }else{
        dScore(x,d) = make_float4(1,0,0,1);
    }
}

void CostVolumeCrossSection(
    Image<float4> dScore, Volume<CostVolElem> dCostVol, int y
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dScore);
    KernCostVolumeCrossSection<<<gridDim,blockDim>>>(dScore, dCostVol.ImageXZ(y));
}

}

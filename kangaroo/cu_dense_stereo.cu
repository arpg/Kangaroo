#include "kangaroo.h"
#include "launch_utils.h"
#include "patch_score.h"
#include "disparity.h"
#include "InvalidValue.h"
//#include "ImageApron.h"

namespace Gpu
{

const int MinDisparity = 1;
const int DefaultRad = 1;
//typedef SSNDPatchScore<float,DefaultRad,ImgAccessRaw> DefaultSafeScoreType;
typedef SANDPatchScore<float,DefaultRad,ImgAccessRaw> DefaultSafeScoreType;
//typedef SinglePixelSqPatchScore<float,ImgAccessRaw> DefaultSafeScoreType;

//////////////////////////////////////////////////////
// Scanline rectified dense stereo
//////////////////////////////////////////////////////

template<typename TD, typename TI, typename Score>
__global__ void KernDenseStereo(
    Image<TD> dDisp, Image<TI> dCamLeft, Image<TI> dCamRight, TD maxDispVal, TD dispStep, float acceptThresh
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    TD bestDisp = InvalidValue<TD>::Value();

    if( Score::width  <= x && x < (dCamLeft.w - Score::width) &&
        Score::height <= y && y < (dCamLeft.h - Score::height) )
    {
        // Search for best matching pixel
        float bestScore = 1E+36;

        TD sndBestDisp = InvalidValue<TD>::Value();
        float sndBestScore = 1E+37;

        TD minDisp = min(maxDispVal, (TD)0);
        TD maxDisp = max((TD)0, maxDispVal);
        minDisp = max((int)minDisp, -(int)( ((int)dCamLeft.w - (int)Score::width) - (int)x));
        maxDisp = min((int)maxDisp, (int)(x + Score::width));

        for(TD c = minDisp; c <= maxDisp; c += dispStep ) {
            const float score =  Score::Score(dCamLeft, x,y, dCamRight, x-c, y);
            if(score < bestScore) {
                sndBestDisp = bestDisp;
                sndBestScore = bestScore;
                bestDisp = c;
                bestScore = score;
            }else if( score <= sndBestScore) {
                sndBestDisp = c;
                sndBestScore = score;
            }
        }
        if(abs(bestDisp-sndBestDisp) > 1) {
            const float cd = (sndBestScore - bestScore) / bestScore;
            if( cd < acceptThresh ) {
                bestDisp = InvalidValue<TD>::Value();
            }
        }
    }

    dDisp(x,y) = bestDisp;
}

template<typename TDisp, typename TImg>
void DenseStereo(
    Image<TDisp> dDisp, const Image<TImg> dCamLeft, const Image<TImg> dCamRight,
    TDisp maxDisp, float acceptThresh, int score_rad
) {
    dim3 blockDim(dDisp.w, 1);
    dim3 gridDim(1, dDisp.h);

    const TDisp dispStep = 1;

    if( score_rad == 0 ) {
        KernDenseStereo<TDisp, TImg, SinglePixelSqPatchScore<float,ImgAccessRaw > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
    }else if(score_rad == 1 ) {
        KernDenseStereo<TDisp, TImg, SANDPatchScore<float,1,ImgAccessRaw > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
    }else if( score_rad == 2 ) {
        KernDenseStereo<TDisp, TImg, SANDPatchScore<float,2,ImgAccessRaw > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
    }else if(score_rad == 3 ) {
        KernDenseStereo<TDisp, TImg, SANDPatchScore<float,3,ImgAccessRaw > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
    }else if( score_rad == 4 ) {
        KernDenseStereo<TDisp, TImg, SANDPatchScore<float,4,ImgAccessRaw > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
    }else if(score_rad == 5 ) {
        KernDenseStereo<TDisp, TImg, SANDPatchScore<float,5,ImgAccessRaw > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
    }else if(score_rad == 6 ) {
        KernDenseStereo<TDisp, TImg, SANDPatchScore<float,6,ImgAccessRaw > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
    }else if(score_rad == 7 ) {
        KernDenseStereo<TDisp, TImg, SANDPatchScore<float,7,ImgAccessRaw > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
    }
}

template void DenseStereo<unsigned char, unsigned char>(Image<unsigned char>, const Image<unsigned char>, const Image<unsigned char>, unsigned char, float, int);
template void DenseStereo<char, unsigned char>(Image<char>, const Image<unsigned char>, const Image<unsigned char>, char, float, int);

void DenseStereoSubpix(
    Image<float> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight, float maxDisp, float dispStep, float acceptThresh, int score_rad, bool score_normed
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dDisp);

    if(score_normed) {
        if( score_rad == 0 ) {
            KernDenseStereo<float, unsigned char, SinglePixelSqPatchScore<float,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if(score_rad == 1 ) {
            KernDenseStereo<float, unsigned char, SANDPatchScore<float,1,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if( score_rad == 2 ) {
            KernDenseStereo<float, unsigned char, SANDPatchScore<float,2,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if(score_rad == 3 ) {
            KernDenseStereo<float, unsigned char, SANDPatchScore<float,3,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if( score_rad == 4 ) {
            KernDenseStereo<float, unsigned char, SANDPatchScore<float,4,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if(score_rad == 5 ) {
            KernDenseStereo<float, unsigned char, SANDPatchScore<float,5,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if(score_rad == 6 ) {
            KernDenseStereo<float, unsigned char, SANDPatchScore<float,6,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if(score_rad == 7 ) {
            KernDenseStereo<float, unsigned char, SANDPatchScore<float,7,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }
    }else{
        if( score_rad == 0 ) {
            KernDenseStereo<float, unsigned char, SinglePixelSqPatchScore<float,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if(score_rad == 1 ) {
            KernDenseStereo<float, unsigned char, SADPatchScore<float,1,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if( score_rad == 2 ) {
            KernDenseStereo<float, unsigned char, SADPatchScore<float,2,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if(score_rad == 3 ) {
            KernDenseStereo<float, unsigned char, SADPatchScore<float,3,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if( score_rad == 4 ) {
            KernDenseStereo<float, unsigned char, SADPatchScore<float,4,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }else if(score_rad == 5 ) {
            KernDenseStereo<float, unsigned char, SADPatchScore<float,5,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
        }
    }
}

//////////////////////////////////////////////////////
// Check computed disparity is a local minima for reverse match
//////////////////////////////////////////////////////

template<typename TD, typename TI, typename Score>
__global__ void KernReverseCheck(
    Image<TD> dDisp, Image<TI> dCamLeft, Image<TI> dCamRight
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const int d = dDisp(x,y);
    const int rx = x - d;

    int best = 0;
    float bestscore = 1E10;

    // Check that this pixel is also a minima for the right image
    const int rad = 10;
#pragma unroll
    for(int i=-rad; i <= rad; ++i) {
        const float s = Score::Score(dCamLeft, x+i,y, dCamRight, rx,y);
        if(s < bestscore) {
            bestscore = s;
            best = i;
        }
    }

    // If not, mark match as invalid
    if(best != 0) {
        dDisp(x,y) = InvalidValue<TD>::Value();
    }
}

void ReverseCheck(
    Image<unsigned char> dDisp, const Image<unsigned char> dCamLeft, const Image<unsigned char> dCamRight
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dDisp);
    KernReverseCheck<unsigned char, unsigned char, DefaultSafeScoreType><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight);
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

    // Ignore things at infinity
    if(bestDisp < MinDisparity) {
        dDispOut(x,y) = InvalidValue<TDo>::Value();
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
        dDispOut(x,y) = InvalidValue<TDo>::Value();
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
    dVbo(u,v) = DepthFromDisparity(u,v, dDisp(u,v), baseline, fu, fv, u0, v0, MinDisparity);
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
//    fu * baseline / d;
    Pv.z = fu / (minz * d);
    Pv.x = Pv.z * (u-u0) / fu;
    Pv.y = Pv.z * (v-v0) / fv;

    const float2 pc = dn(KT_cv * Pv);

    if( dimgc.InBounds(pc.x, pc.y,5) ) {
//        vol(u,v,d) = 1.0f;
        const float score =  Score::Score(dimgv, u,v, dimgc, pc.x, pc.y) / (float)(Score::area);
//        const float score = (dimgv(u,v) - dimgc.template GetBilinear<float>(pc)) / 255.0f;
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
    KernAddToCostVolume<unsigned char, SSNDPatchScore<float,DefaultRad,ImgAccessBilinearClamped<float> > ><<<gridDim,blockDim>>>(dvol,dimgv,dimgc, KT_cv, fu,fv,u0,v0, minz,maxz, levels);
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

//////////////////////////////////////////////////////

template<typename To, typename Ti>
__global__ void KernFilterDispGrad(Image<To> dOut, Image<Ti> dIn, float threshold )
{
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    const float dx = dOut.template GetCentralDiffDx<float>(x,y);
    const float dy = dOut.template GetCentralDiffDy<float>(x,y);
    const bool valid = dx*dx + dy*dy < threshold;

    dOut(x,y) = valid ? dIn(x,y) : -1;
}

void FilterDispGrad(
    Image<float> dOut, Image<float> dIn, float threshold
) {
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dOut, 16, 16);
    KernFilterDispGrad<float,float><<<gridDim,blockDim>>>(dOut, dIn, threshold);
}


}

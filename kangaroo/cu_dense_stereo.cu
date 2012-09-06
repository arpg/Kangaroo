#include "kangaroo.h"
#include "launch_utils.h"
#include "patch_score.h"
#include "disparity.h"
#include "InvalidValue.h"
#include "ImageApron.h"

namespace Gpu
{

const int MinDisparity = 1;
const int DefaultRad = 2;
//typedef SSNDPatchScore<float,DefaultRad,ImgAccessRaw> DefaultSafeScoreType;
typedef SANDPatchScore<float,DefaultRad,ImgAccessRaw> DefaultSafeScoreType;
//typedef SinglePixelSqPatchScore<float,ImgAccessRaw> DefaultSafeScoreType;

//////////////////////////////////////////////////////
// Cost Volume minimum
//////////////////////////////////////////////////////

template<typename Tdisp, typename Tvol>
__global__ void KernCostVolMinimum(Image<Tdisp> disp, Volume<Tvol> vol, unsigned maxDispVal)
{
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    Tdisp bestd = 0;
    Tvol bestc = vol(x,y,0);

    const int maxDisp = min(maxDispVal, x+1);
    for(unsigned d=1; d < maxDisp; ++d) {
        const Tvol c = vol(x,y,d);
        if(c < bestc) {
            bestc = c;
            bestd = d;
        }
    }
    disp(x,y) = bestd;
}


template<typename Tdisp, typename Tvol>
void CostVolMinimum(Image<Tdisp> disp, Volume<Tvol> vol, unsigned maxDisp)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim,disp);
    KernCostVolMinimum<Tdisp,Tvol><<<gridDim,blockDim>>>(disp,vol,maxDisp);
}

template void CostVolMinimum<>(Image<char>,Volume<float>,unsigned);
template void CostVolMinimum<>(Image<char>,Volume<int>,unsigned);
template void CostVolMinimum<>(Image<char>,Volume<unsigned int>,unsigned);
template void CostVolMinimum<>(Image<char>,Volume<unsigned short>,unsigned);
template void CostVolMinimum<>(Image<char>,Volume<unsigned char>,unsigned);
template void CostVolMinimum<>(Image<float>,Volume<float>,unsigned);
template void CostVolMinimum<>(Image<float>,Volume<unsigned short>,unsigned);


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

const int MAXBW = 512;

//template<typename TD, typename TI, typename Score>
//__global__ void KernDenseStereo(
//    Image<TD> dDisp, Image<TI> dCamLeft, Image<TI> dCamRight, TD maxDispVal, TD dispStep, float acceptThresh
//) {
//    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
//    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

//    const int W = Score::width;
//    const int RAD = W / 2;

////    TI patch[W*W];

//    // only enough shared mem to cache right image
////    __shared__ ImageApronRows<TI,MAXBW,1,RAD> apron_l;
//    __shared__ ImageApronRows<TI,MAXBW,1,RAD> apron_r;
////    __shared__ ImageApronRows<TI,MAXBW,1,0> col_avg_l;
////    __shared__ ImageApronRows<TI,MAXBW,1,0> col_avg_r;
////    __shared__ ImageApronRows<TI,MAXBW,1,0> avg_l;
////    __shared__ ImageApronRows<TI,MAXBW,1,0> avg_r;

//    ///////////////////////////////////
////    // Cache line of right/left image +/- RAD
////    apron_l.CacheImage(dCamLeft);
//    apron_r.CacheImage(dCamRight);

////    __syncthreads();

////    ///////////////////////////////////
////    // Cache sum of colums for norm
//////    int colsuml = 0;
////    int colsumr = 0;
////#pragma unroll
////    for(int i=-RAD; i<=RAD; ++i) {
//////        colsuml += apron_l.GetRelThread(0,i);
////        colsumr += apron_r.GetRelThread(0,i);
////    }
//////    col_avg_l.GetRelThread(0,0) = colsuml / W;
////    col_avg_r.GetRelThread(0,0) = colsumr / W;
////    __syncthreads();

////    ///////////////////////////////////
////    // Cache sum of block for norm
//////    int suml = 0;
////    int sumr = 0;
////#pragma unroll
////    for(int i=-RAD; i<=RAD; ++i) {
//////        suml += col_avg_l.GetRelThreadClampX(i,0);
////        sumr += col_avg_r.GetRelThreadClampX(i,0);
////    }
//////    avg_l.GetRelThread(0,0) = suml / W;
////    avg_r.GetRelThread(0,0) = sumr / W;

//    ///////////////////////////////////
//    // Cache left patch, compute mean
//////    int sum_l = 0;
////    for(int r=-RAD; r<= RAD; ++r) {
////#pragma unroll
////        for(int c=-RAD; c<=RAD; ++c) {
////            const TI val = dCamLeft.GetWithClampedRange(x+c, y+r);
////            patch[(RAD+r)*W+(RAD+c)] = val;
//////            sum_l += val;
////        }
////    }
////    const TI avg_l = sum_l / (W*W);

//    __syncthreads();

//    TD bestDisp = InvalidValue<TD>::Value();

//    if( maxDispVal+Score::width <= x && x < (dCamLeft.w - Score::width) &&
//        Score::height <= y && y < (dCamLeft.h - Score::height) )
//    {
//        // Search for best matching pixel
//        float bestScore = 1E+36;

////        TD sndBestDisp = InvalidValue<TD>::Value();
////        float sndBestScore = 1E+37;

////        TD minDisp = min(maxDispVal, (TD)0);
////        TD maxDisp = max((TD)0, maxDispVal);
////        minDisp = max((int)minDisp, -(int)( ((int)dCamLeft.w - (int)Score::width) - (int)x));
////        maxDisp = min((int)maxDisp, (int)(x + Score::width));

//        for(TD c = 0; c <= maxDispVal; c += 1 ) {
//            float score = 0;

//            for(int ky=-RAD; ky <= RAD; ++ky ) {
//#pragma unroll
//                for(int kx=-RAD; kx <= RAD; ++kx ) {
////                    const int pl = apron_l.GetRelThread(kx,ky);
//                    const int pl = 0;//patch[(RAD+ky)*W+(RAD+kx)];
//                    const int pr = apron_r.GetRelThread(kx-c,ky);
//                    score += abs(pl - pr);
//                }
//            }

//////            Score::Score(dCamLeft, x,y, dCamRight, x-c, y);
//            if(score < bestScore) {
////                sndBestDisp = bestDisp;
////                sndBestScore = bestScore;
//                bestDisp = c;
//                bestScore = score;
////            }else if( score <= sndBestScore) {
////                sndBestDisp = c;
////                sndBestScore = score;
//            }
//        }
////        if(abs(bestDisp-sndBestDisp) > 1) {
////            const float cd = (sndBestScore - bestScore) / bestScore;
////            if( cd < acceptThresh ) {
////                bestDisp = InvalidValue<TD>::Value();
////            }
////        }
//    }

//    dDisp(x,y) = bestDisp;
//}

template<typename TDisp, typename TImg>
void DenseStereo(
    Image<TDisp> dDisp, const Image<TImg> dCamLeft, const Image<TImg> dCamRight,
    TDisp maxDisp, float acceptThresh, int score_rad
) {
    dim3 blockDim(dDisp.w, 1);
    dim3 gridDim(1, dDisp.h);
//    InitDimFromOutputImageOver(blockDim,gridDim,dDisp);

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

//    if(score_normed) {
//        if( score_rad == 0 ) {
//            KernDenseStereo<float, unsigned char, SinglePixelSqPatchScore<float,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if(score_rad == 1 ) {
//            KernDenseStereo<float, unsigned char, SANDPatchScore<float,1,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if( score_rad == 2 ) {
//            KernDenseStereo<float, unsigned char, SANDPatchScore<float,2,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if(score_rad == 3 ) {
//            KernDenseStereo<float, unsigned char, SANDPatchScore<float,3,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if( score_rad == 4 ) {
//            KernDenseStereo<float, unsigned char, SANDPatchScore<float,4,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if(score_rad == 5 ) {
//            KernDenseStereo<float, unsigned char, SANDPatchScore<float,5,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if(score_rad == 6 ) {
//            KernDenseStereo<float, unsigned char, SANDPatchScore<float,6,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if(score_rad == 7 ) {
//            KernDenseStereo<float, unsigned char, SANDPatchScore<float,7,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }
//    }else{
//        if( score_rad == 0 ) {
//            KernDenseStereo<float, unsigned char, SinglePixelSqPatchScore<float,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if(score_rad == 1 ) {
//            KernDenseStereo<float, unsigned char, SADPatchScore<float,1,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if( score_rad == 2 ) {
//            KernDenseStereo<float, unsigned char, SADPatchScore<float,2,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if(score_rad == 3 ) {
//            KernDenseStereo<float, unsigned char, SADPatchScore<float,3,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if( score_rad == 4 ) {
//            KernDenseStereo<float, unsigned char, SADPatchScore<float,4,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }else if(score_rad == 5 ) {
//            KernDenseStereo<float, unsigned char, SADPatchScore<float,5,ImgAccessBilinear<float> > ><<<gridDim,blockDim>>>(dDisp, dCamLeft, dCamRight, maxDisp, dispStep, acceptThresh);
//        }
//    }
}

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////

const int RAD = 3;
const int W = 2*RAD+1;

__global__ void KernDenseStereoTest(
    Image<float> dDisp, Image<unsigned char> dCamLeft, Image<unsigned char> dCamRight, int maxDisp
) {
    const uint x = threadIdx.x;
    const uint y = blockIdx.y;

    __shared__ unsigned char cache_l[W][MAXBW];
    __shared__ unsigned char cache_r[W][MAXBW+1];

#pragma unroll
    for(int r=0; r<W; ++r ) {
        cache_l[r][x] = dCamLeft.Get(x,y+r-RAD);
        cache_r[r][x] = dCamRight.Get(x,y+r-RAD);
    }

    __syncthreads();

    int bestScore = 0xFFFFF;
    int bestDisp = 0;

    const int maxClipDisp = min(x-RAD,maxDisp);
    for(int d=0; d<maxClipDisp; ++d)
    {
        const int xd = x-d;
        int score = 0;
#pragma unroll
        for(int r=0; r<W; ++r) {
            score += abs(cache_l[r][x] - cache_r[r][xd]);
//            const int yr = y-RAD+r;
//            score += abs(dCamLeft(x,yr) - dCamRight(xd,yr));
        }

        if(score < bestScore) {
            bestScore = score;
            bestDisp = d;
        }
    }

    dDisp(x,y) = bestDisp;
}

void DenseStereoTest(
    Image<float> dDisp, Image<unsigned char> dCamLeft, Image<unsigned char> dCamRight, int maxDisp
) {
    const int w = dDisp.w;
    const int h = dDisp.h - 2*RAD;
    const int x = 0;
    const int y = RAD;

    dim3 blockDim(w, 1);
    dim3 gridDim(1, h);
    KernDenseStereoTest<<<gridDim,blockDim>>>(dDisp.SubImage(x,y,w,h), dCamLeft.SubImage(x,y,w,h), dCamRight.SubImage(x,y,w,h), maxDisp);
}

//////////////////////////////////////////////////////
// Check Left and Right disparity images match
//////////////////////////////////////////////////////

template<typename TD>
__global__ void KernLeftRightCheck(
    Image<TD> dispL, Image<TD> dispR, int maxDiff
) {
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if( dispL.InBounds(x,y) ) {
        const TD dl = dispL(x,y);

        if( 0 <= ((int)x-(int)dl) ) {
            const TD dr = dispR(x - dl, y);
            if(abs(dl - (-dr)) > maxDiff) {
                dispL(x,y) = InvalidValue<TD>::Value();
            }
        }else{
            dispL(x,y) = InvalidValue<TD>::Value();
        }
    }
}

void LeftRightCheck(Image<char> dispL, Image<char> dispR, int maxDiff)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim,gridDim, dispL);
    KernLeftRightCheck<char><<<gridDim,blockDim>>>(dispL, dispR, maxDiff);
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

void CostVolumeZero(Volume<CostVolElem> costvol )
{
    CostVolElem initial;
    initial.sum = 0;
    initial.n = 0;
    costvol.Fill(initial);
}

//////////////////////////////////////////////////////

template<typename TD, typename TI, typename Score>
__global__ void KernCostVolumeFromStereo(
    Volume<CostVolElem> dvol, Image<TI> dimgl, Image<TI> dimgr
) {
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const int d = blockIdx.z*blockDim.z + threadIdx.z;

    if( u-d >= (int)Score::rad) {
        CostVolElem elem;
        elem.sum = Score::Score(dimgl, u,v, dimgr, u-d, v) / Score::area;
        elem.n = 1;
        dvol(u,v,d) = elem;
    }
}

void CostVolumeFromStereo(Volume<CostVolElem> dvol, Image<unsigned char> dimgl, Image<unsigned char> dimgr )
{
    dim3 blockDim(8,8,8);
    dim3 gridDim(dvol.w / blockDim.x, dvol.h / blockDim.y, dvol.d / blockDim.z);
    KernCostVolumeFromStereo<unsigned char, unsigned char, DefaultSafeScoreType><<<gridDim,blockDim>>>(dvol,dimgl,dimgr);
}

//////////////////////////////////////////////////////

template<typename TI, typename Score>
__global__ void KernAddToCostVolume(
    Volume<CostVolElem> dvol, const Image<TI> dimgv,
    const Image<TI> dimgc, Mat<float,3,4> KT_cv,
    float fu, float fv, float u0, float v0,
    float baseline
){
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const int d = blockIdx.z*blockDim.z + threadIdx.z;

    float3 Pv;
    Pv.z = fu * baseline / d;
    Pv.x = Pv.z * (u-u0) / fu;
    Pv.y = Pv.z * (v-v0) / fv;

    const float3 KPc = KT_cv * Pv;
    const float2 pc = dn(KPc);

    if( KPc.z > 0 && dimgc.InBounds(pc.x, pc.y,5) ) {
//        vol(u,v,d) = 1.0f;
        const float score =  Score::Score(dimgv, u,v, dimgc, pc.x, pc.y) / (float)(Score::area);
//        const float score = (dimgv(u,v) - dimgc.template GetBilinear<float>(pc)) / 255.0f;
        CostVolElem elem = dvol(u,v,d);
        elem.sum += score;
        elem.n += 1;
        dvol(u,v,d) = elem;
    }
}

void CostVolumeAdd(Volume<CostVolElem> dvol, const Image<unsigned char> dimgv,
    const Image<unsigned char> dimgc, Mat<float,3,4> KT_cv,
    float fu, float fv, float u0, float v0,
    float baseline, int levels
) {
    dim3 blockDim(8,8,8);
    dim3 gridDim(dvol.w / blockDim.x, dvol.h / blockDim.y, dvol.d / blockDim.z);
    KernAddToCostVolume<unsigned char, SANDPatchScore<float,DefaultRad,ImgAccessBilinearClamped<float> > ><<<gridDim,blockDim>>>(dvol,dimgv,dimgc, KT_cv, fu,fv,u0,v0, baseline);
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

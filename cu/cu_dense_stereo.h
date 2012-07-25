#pragma once

#include "all.h"

#include "cu_patch_score.h"

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
    Image<float4> dVbo, const Image<float> dDisp, double baseline, double fu, double fv, double u0, double v0
) {
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const float invalid = 0.0f/0.0f;

    const float disp = dDisp(u,v);
    const float z = disp > 2 ? fu * baseline / -disp : invalid;

    // (x,y,1) = kinv * (u,v,1)'
    const float x = -z * (u-u0) / fu;
    const float y = z * (v-v0) / fv;

    dVbo(u,v) = make_float4(x,y,z,1);
}

void DisparityImageToVbo(Image<float4> dVbo, const Image<float> dDisp, double baseline, double fu, double fv, double u0, double v0)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dVbo);
    KernDisparityImageToVbo<<<gridDim,blockDim>>>(dVbo, dDisp, baseline, fu, fv, u0, v0);
}

//////////////////////////////////////////////////////
// Kinect depthmap to vertex array
//////////////////////////////////////////////////////

__global__ void KernKinectToVbo(
    Image<float4> dVbo, const Image<unsigned short> dKinectDepth, double fu, double fv, double u0, double v0
) {
    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const float invalid = 0.0f / 0.0f;

    const float kd = dKinectDepth(u,v);
    const float z = (kd > 0) ? -kd / 1000.0 : invalid;

    // (x,y,1) = kinv * (u,v,1)'
    const float x = -z * (u-u0) / fu;
    const float y = z * (v-v0) / fv;

    dVbo(u,v) = make_float4(x,y,z,1);
}

void KinectToVbo(Image<float4> dVbo, const Image<unsigned short> dKinectDepth, double fu, double fv, double u0, double v0)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dVbo);
    KernKinectToVbo<<<gridDim,blockDim>>>(dVbo, dKinectDepth, fu, fv, u0, v0);
}

//////////////////////////////////////////////////////
// Make Index Buffer for rendering
//////////////////////////////////////////////////////

__global__ void KernGenerateTriangleStripIndexBuffer(Image<uint2> dIbo)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    const unsigned int pixIndex = y*dIbo.w + x;
    dIbo(x,y) = make_uint2(pixIndex, pixIndex + dIbo.w);
}

void GenerateTriangleStripIndexBuffer( Image<uint2> dIbo)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim,gridDim, dIbo);
    KernGenerateTriangleStripIndexBuffer<<<gridDim,blockDim>>>(dIbo);
}

//////////////////////////////////////////////////////
// Plane Fitting
//////////////////////////////////////////////////////

__global__ void KernPlaneFitGN(const Image<float4> dVbo, const Mat<float,3,3> Qinv, const Mat<float,3> zhat, Image<LeastSquaresSystem<float,3> > dSum, Image<float> dErr, float within, float c )
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    const float4 P = dVbo(x,y);
    LeastSquaresSystem<float,3> sum;

    if( length(P) < within ) {
        const Mat<float,3,1> nhat = Qinv * zhat;
        const float dinv = sqrt(nhat * nhat);
        const float d = 1.0 / dinv;

        const float np_p1 = nhat[0] * P.x + nhat[1] * P.y + nhat[2] * P.z + 1;
        const float y = d * np_p1;
        const float absy = abs(y);
        const float roc = y/c;
        const float omrocsq = (1-roc*roc);
        const float w = (absy <= c) ? omrocsq*omrocsq : 0;

        const Mat<float,3,1> dn_dz0 = zhat[0] * (Mat<float,3,1>){Qinv(0,0), Qinv(1,0), Qinv(2,0)};
        const Mat<float,3,1> dn_dz1 = zhat[1] * (Mat<float,3,1>){Qinv(0,1), Qinv(1,1), Qinv(2,1)};
        const Mat<float,3,1> dn_dz2 = zhat[2] * (Mat<float,3,1>){Qinv(0,2), Qinv(1,2), Qinv(2,2)};

        Mat<float,1,3> Ji;
        Ji[0] = ((-d*d*d*np_p1) * (nhat * dn_dz0)) + d * (dn_dz0[0]*P.x + dn_dz0[1]*P.y + dn_dz0[2]*P.z);
        Ji[1] = ((-d*d*d*np_p1) * (nhat * dn_dz1)) + d * (dn_dz1[0]*P.x + dn_dz1[1]*P.y + dn_dz1[2]*P.z);
        Ji[2] = ((-d*d*d*np_p1) * (nhat * dn_dz2)) + d * (dn_dz2[0]*P.x + dn_dz2[1]*P.y + dn_dz2[2]*P.z);

        sum.JTJ = OuterProduct(Ji, w);
        sum.JTy = Ji * (y * w);
        sum.sqErr = y*y;
        sum.obs = 1;
    }else{
        sum.SetZero();
    }

    dErr(x,y) = sum.sqErr;
    dSum(x,y) = sum;
}

LeastSquaresSystem<float,3> PlaneFitGN(const Image<float4> dVbo, const Mat<float,3,3> Qinv, const Mat<float,3> zhat, Image<unsigned char> dWorkspace, Image<float> dErr, float within, float c )
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImage(blockDim, gridDim, dVbo);
    Image<LeastSquaresSystem<float,3> > dSum = dWorkspace.PackedImage<LeastSquaresSystem<float,3> >(dVbo.w, dVbo.h);

    KernPlaneFitGN<<<gridDim,blockDim>>>(dVbo, Qinv, zhat, dSum, dErr, within, c );

    LeastSquaresSystem<float,3> sum;
    sum.SetZero();

    return thrust::reduce(dSum.begin(), dSum.end(), sum, thrust::plus<LeastSquaresSystem<float,3> >() );
}

}

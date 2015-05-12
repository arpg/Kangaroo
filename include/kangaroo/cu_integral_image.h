#pragma once

#include <kangaroo/platform.h>
#include <kangaroo/Image.h>
#include <kangaroo/cu_operations.h>

namespace roo
{

template<typename Tout, typename Tin>
KANGAROO_EXPORT
void Transpose(Image<Tout> out, Image<Tin> in);

template<typename Tout, typename Tin>
KANGAROO_EXPORT
void PrefixSumRows(Image<Tout> out, Image<Tin> in);

//////////////////////////////////////////////////////

template<typename Tout, typename Tin>
KANGAROO_EXPORT
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

}

#include "kangaroo.h"
#include "launch_utils.h"
#include "patch_score.h"
#include "disparity.h"
#include "InvalidValue.h"
#include "ImageApron.h"

namespace Gpu
{

template<typename TH, typename TC>
__global__ void KernSemiGlobalMatching(Volume<TH> volH, Volume<TC> volC, Image<unsigned char> left, int maxDispVal, float P1, float P2, int xoffset, int yoffset, int dx, int dy, unsigned pathlen)
{
    int x = xoffset + threadIdx.x;
    int y = yoffset + threadIdx.y;

    TH lastBestCr = 0;
    int last_c = left(x,y);

    const int maxDisp = min(maxDispVal,x+1);
    int lastMaxDisp = maxDisp;
    for(int d=0; d < maxDisp; ++d) {
        volH(x,y,d) += volC(x,y,d);
    }

    x += dx;
    y += dy;

    for(int r=1; r<pathlen; ++r)
    {
        const int c = left(x,y);
        const float __P2 = P2 / (float)(1+abs(last_c-c));
        const float _P2 = max(P1+1.0f,__P2);
        TH bestCr = 0xFFFF;
        const int maxDisp = min(maxDispVal,x+1);
        for(int d=0; d < maxDisp; ++d) {
            const TH C = volC(x,y,d);
            const TH Crp = d < lastMaxDisp ? volH(x-dx,y-dy,d) : 0xEFFFFFFF;
            const TH Crpm = d>0 ? volH(x-dx,y-dy,d-1) : 0xEFFFFFFF;
            const TH Crpp = (d+1) < (lastMaxDisp) ? volH(x-dx,y-dy,d+1) : 0xEFFFFFFF;
            const TH Crmin = lastBestCr;
            const TH Cr = C +  min(Crp,min(min(Crpm,Crpp)+P1,Crmin+_P2)) - Crmin;
            bestCr = min(bestCr, Cr);
            volH(x,y,d) += Cr;
        }
        x += dx;
        y += dy;
        lastBestCr = bestCr;
        last_c = c;
        lastMaxDisp = maxDisp;
    }

}

void SemiGlobalMatching(Volume<float> volH, Volume<unsigned char> volC, Image<unsigned char> left, int maxDisp, float P1, float P2, bool dohoriz, bool dovert, bool doreverse)
{
    volH.Memset(0);
    dim3 blockDim(volC.w, 1);
    dim3 gridDim(1, 1);
    if(dovert) {
        KernSemiGlobalMatching<<<gridDim,blockDim>>>(volH,volC,left,maxDisp,P1,P2,0,0,0,1,volC.h);
        if(doreverse) {
            KernSemiGlobalMatching<<<gridDim,blockDim>>>(volH,volC,left,maxDisp,P1,P2,0,volC.h-1,0,-1,volC.h);
        }
    }

    if(dohoriz) {
        dim3 blockDim2(1, volC.h);
        KernSemiGlobalMatching<<<gridDim,blockDim2>>>(volH,volC,left,maxDisp,P1,P2,0,0,1,0,volC.w);
        if(doreverse) {
            KernSemiGlobalMatching<<<gridDim,blockDim2>>>(volH,volC,left,maxDisp,P1,P2,volC.w-1,0,-1,0,volC.w);
        }
    }
}

}

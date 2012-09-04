#include "kangaroo.h"
#include "launch_utils.h"
#include "patch_score.h"
#include "disparity.h"
#include "InvalidValue.h"
#include "ImageApron.h"

namespace Gpu
{

template<typename TH, typename TC>
__global__ void KernSemiGlobalMatching(Volume<TH> volH, Volume<TC> volC, int maxDisp, unsigned P1, unsigned P2, int xoffset, int yoffset, int dx, int dy, unsigned pathlen)
{
    int x = xoffset + threadIdx.x;
    int y = yoffset + threadIdx.y;

    TH lastBestCr = 0;

    for(int d=0; d < maxDisp; ++d) {
        volH(x,y,d) += volC(x,y,d);
    }

    x += dx;
    y += dy;

    for(int r=1; r<pathlen; ++r)
    {
        TH bestCr = 0xFFFF;
        for(int d=0; d < maxDisp; ++d) {
            const TH C = volC(x,y,d);
            const TH Crp = volH(x-dx,y-dy,d);
            const TH Crpm = d>0 ? volH(x-dx,y-dy,d-1) : 0xFFFF;
            const TH Crpp = d<(maxDisp-1) ? volH(x-dx,y-dy,d+1) : 0xFFFF;
            const TH Crmin = lastBestCr;
            const TH Cr = C +  min(Crp,min(min(Crpm,Crpp)+P1,Crmin+P2)) - Crmin;
            bestCr = min(bestCr, Cr);
            volH(x,y,d) += Cr;
        }
        x += dx;
        y += dy;
        lastBestCr = bestCr;
    }

}

void SemiGlobalMatching(Volume<int> volH, Volume<unsigned char> volC, int maxDisp, unsigned P1, unsigned P2)
{
    volH.Memset(0);
    dim3 blockDim(volC.w, 1);
    dim3 gridDim(1, 1);
    KernSemiGlobalMatching<int,unsigned char><<<gridDim,blockDim>>>(volH,volC,maxDisp,P1,P2,0,0,0,1,volC.h);
    KernSemiGlobalMatching<int,unsigned char><<<gridDim,blockDim>>>(volH,volC,maxDisp,P1,P2,0,volC.h-1,0,-1,volC.h);

    dim3 blockDim2(1, volC.h);
    KernSemiGlobalMatching<<<gridDim,blockDim2>>>(volH,volC,maxDisp,P1,P2,0,0,1,0,volC.w);
    KernSemiGlobalMatching<<<gridDim,blockDim2>>>(volH,volC,maxDisp,P1,P2,volC.w-1,0,-1,0,volC.w);
}

}

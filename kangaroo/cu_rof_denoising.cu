#include "kangaroo.h"
#include "variational.h"
#include "launch_utils.h"

using namespace std;

namespace Gpu
{

//////////////////////////////////////////////////////
// Divergence, Div A
// da/dx + da/dy
//////////////////////////////////////////////////////

inline __host__ __device__
float DivA(const Image<float2>& A, int x, int y)
{
    float divA = 0;

    // TODO: Stop doing extra deriv

    if(x>0) {
        const float2 dAdx = A.GetBackwardDiffDx<float2>(x,y);
        divA += dAdx.x;
    }

    if(y>0) {
        const float2 dAdy = A.GetBackwardDiffDy<float2>(x,y);
        divA += dAdy.y;
    }

    return divA;
}

__global__ void KernDivergence(Image<float> divA, Image<float2> A)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < A.w && y < A.h ) {
        const float diva = DivA(A,x,y);
        divA(x,y) =  diva;
    }
}

void Divergence(Image<float> divA, Image<float2> A)
{
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, A);
    KernDivergence<<<gridDim,blockDim>>>(divA,A);
}

//////////////////////////////////////////////////////
// ROF p ascent
//////////////////////////////////////////////////////

__global__ void KernDenoisingRof_pAscent(
        Image<float2> p, const Image<float> u,
        float sigma
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < u.w && y < u.h ) {
        float2 du = make_float2(0,0);

        if(x < u.w-1 ) {
            du.x = u(x+1,y) - u(x,y);
        }

        if(y < u.h-1 ) {
            du.y = u(x,y+1) - u(x,y);
        }

        const float2 np = p(x,y) + sigma * du;
        const float mag_np = sqrt(np.x*np.x + np.y*np.y);
        const float reprojection = max(1.0f, mag_np);
        p(x,y) = np / reprojection;
    }
}

void DenoisingRof_pAscent(
        Image<float2> p, const Image<float> u,
        float sigma, Image<unsigned char> scratch
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, u);
    KernDenoisingRof_pAscent<<<gridDim,blockDim>>>(p,u,sigma);
//    const float sumabs = ImageL1<float, float2>(p, scratch);
//    cout << sumabs << endl;
//    ElementwiseScaleBias<float2,float2,float2>(p, p, 1.0f / max(1.0f,sumabs), make_float2(0,0) );
}

//////////////////////////////////////////////////////
// ROF u descent
//////////////////////////////////////////////////////

__global__ void KernDenoisingRof_uDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg,
        float tau, float lambda
) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < imgu.w && y < imgu.h ) {
        float dxp = 0;
        float dyp = 0;
        if ( x > 0 ) dxp = imgp(x,y).x - imgp(x-1,y).x;
        if ( y > 0 ) dyp = imgp(x,y).y - imgp(x,y-1).y;
        const float divp_np1 = dxp + dyp;
//        const float divp_np1 = DivA(imgp,x,y);

        const float g = imgg(x,y);
        const float u_n = imgu(x,y);
        const float u_np1 = (u_n + tau * (divp_np1 + lambda * g)) / (1 + tau*lambda);
        imgu(x,y) = u_np1;
    }
}

void DenoisingRof_uDescent(
        Image<float> imgu, const Image<float2> imgp, const Image<float> imgg,
        float tau, float lambda
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, imgu);
    KernDenoisingRof_uDescent<<<gridDim,blockDim>>>(imgu,imgp,imgg, tau, lambda);
}


} // namespace Gpu

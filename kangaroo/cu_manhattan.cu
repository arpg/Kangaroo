#include "cu_manhattan.h"

#include "LeastSquareSum.h"
#include "launch_utils.h"
#include "MatUtils.h"

namespace roo
{

template<typename T>
__global__ void KernManhattanLineCost(
    Image<float4> out, Image<float4> out2, const Image<T> in,
    Mat<float,3,3> Rhat, float fu, float fv, float u0, float v0,
    float cut, float scale, float min_grad,
    Image<LeastSquaresSystem<float,3> > dsum
){
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    __shared__ SumLeastSquaresSystem<float,3,32,32> sumlss;
    LeastSquaresSystem<float,3>& sum = sumlss.ZeroThisObs();

    if(in.InBounds(x,y))
    {
        out(x,y) = make_float4(0,0,0,1.0f);
        out2(x,y) = make_float4(0,0,0,1);
    }

    const int border=3;
    if(in.InBounds(x,y,border)) {
//        const float dx = in.template GetCentralDiffDx<float>(x,y) / 255.0;
//        const float dy = in.template GetCentralDiffDy<float>(x,y) / 255.0;

        // http://www.holoborodko.com/pavel/image-processing/edge-detection/
        const float dx = (
                    1*in(x+2,y-1) + 2*in(x+1,y-1) - 2*in(x-1,y-1) - 1*in(x-2,y-1) +
                    2*in(x+2,y) + 4*in(x+1,y) - 4*in(x-1,y) - 2*in(x-2,y) +
                    1*in(x+2,y+1) + 2*in(x+1,y+1) - 2*in(x-1,y+1) - 1*in(x-2,y+1)
                    ) / (32.0f * 255.0f);

        const float dy = (
                    1*in(x-1,y+2) + 2*in(x-1,y+1) - 2*in(x-1,y-1) - 1*in(x-1,y-2) +
                    2*in(x,y+2) + 4*in(x,y+1) - 4*in(x,y-1) - 2*in(x,y-2) +
                    1*in(x+1,y+2) + 2*in(x+1,y+1) - 2*in(x+1,y-1) - 1*in(x+1,y-2)
                    ) / (32.0f * 255.0f);

        const float mag = sqrt(dx*dx + dy*dy);

        if( mag > min_grad)
        {
            sum.obs = 1;

            const float3 line = make_float3(-dy,dx,0);
            const float3 ray  = make_float3( (x-u0)/fu, (y-v0)/fv, 1);
            const float3 n = cross(line,ray);
            const float3 m = n / sqrt(dot(n,n));

            const float dotx = Rhat(0,0) * m.x + Rhat(0,1) * m.y + Rhat(0,2) * m.z;
            const float doty = Rhat(1,0) * m.x + Rhat(1,1) * m.y + Rhat(1,2) * m.z;
            const float dotz = Rhat(2,0) * m.x + Rhat(2,1) * m.y + Rhat(2,2) * m.z;
            const float dotxx = dotx*dotx;
            const float dotyy = doty*doty;
            const float dotzz = dotz*dotz;

            const Mat<float,3> dRRm0 = Rhat * make_mat(0,m.z,-m.y);
            const Mat<float,3> dRRm1 = Rhat * make_mat(-m.z,0,m.x);
            const Mat<float,3> dRRm2 = Rhat * make_mat(m.y,-m.x,0);

            const float4 cutcolor = make_float4(0,0,0,1);

            Mat<float,3> J = make_mat(0.0f,0.0f,0.0f);
            float f = 0.0f;

			float4 d1 = {0,0,0,1};
			float4 d2 = {0,0,0,1};

            // categorise pixel as lowest cost line.
            if( dotxx < cut*min(dotyy,dotzz) ) {
                // xline
                f = mag*dotx;
                J = make_mat(mag*dRRm0(0),mag*dRRm1(0),mag*dRRm2(0));
				if( f*f < cut*cut ) {
				    d1.x = scale*f*f;
				    d2.x = 10*mag;
				}
            }else if( dotyy < cut*min(dotxx,dotzz) ) {
                // yline
                f = mag*doty;
                J = make_mat(mag*dRRm0(1),mag*dRRm1(1),mag*dRRm2(1));
				if( f*f < cut*cut ) {
				    d1.y = scale*f*f;
				    d2.y = 10*mag;
				}
            }else if( dotzz < cut*min(dotxx,dotyy) ) {
                // zline
                f = mag*dotz;
                J = make_mat(mag*dRRm0(2),mag*dRRm1(2),mag*dRRm2(2));
				if( f*f < cut*cut ) {
				    d1.z = scale*f*f;
				    d2.z = 10*mag;
				}
            }

			out(x,y) = d1;
			out2(x,y) = d2;

//            if( f*f < cut ) {
                sum.JTy = J*f;
                sum.JTJ = OuterProduct(J);
//            }
        }

//        out2(x,y) = (float4){0.5+dx,0.5+dy,0.5,1};

    }

    sumlss.ReducePutBlock(dsum);
}

LeastSquaresSystem<float,3> ManhattanLineCost(
    Image<float4> out, Image<float4> out2, const Image<unsigned char> in,
    Mat<float,3,3> Rhat, float fu, float fv, float u0, float v0,
    float cut, float scale, float min_grad,
    Image<unsigned char> dWorkspace
) {
    dim3 gridDim, blockDim;
    InitDimFromOutputImageOver(blockDim,gridDim, out);
    HostSumLeastSquaresSystem<float,3> lss(dWorkspace, blockDim, gridDim);
    KernManhattanLineCost<unsigned char><<<gridDim,blockDim>>>(out,out2,in,Rhat,fu,fv,u0,v0,cut,scale,min_grad,lss.LeastSquareImage() );
    return lss.FinalSystem();
}

}

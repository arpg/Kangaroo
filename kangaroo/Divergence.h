#pragma once

#include "Image.h"

namespace Gpu {

//////////////////////////////////////////////////////
// Normalisation of Dual variables back to inside unit ball
//////////////////////////////////////////////////////

inline __host__ __device__
float ProjectUnitBall(float val, float maxrad = 1.0f)
{
    return val / fmaxf(1.0f, abs(val) / maxrad );
}

inline __host__ __device__
float2 ProjectUnitBall(float2 val, float maxrad = 1.0f)
{
    return val / fmaxf(1.0f, sqrt(val.x*val.x + val.y*val.y) / maxrad );
}

inline __host__ __device__
float3 ProjectUnitBall(float3 val, float maxrad = 1.0f)
{
    return val / fmaxf(1.0f, sqrt(val.x*val.x + val.y*val.y + val.z*val.z)  / maxrad );
}

inline __host__ __device__
float4 ProjectUnitBall(float4 val, float maxrad = 1.0f)
{
    return val / fmaxf(1.0f, sqrt(val.x*val.x + val.y*val.y + val.z*val.z + val.w*val.w)  / maxrad );
}

//////////////////////////////////////////////////////
// Gradient of u
//////////////////////////////////////////////////////

inline __host__ __device__
float2 GradUFwd(const Image<float>& imgu, float u, int x, int y)
{
    float2 du = make_float2(0,0);
    if(x < imgu.w-1 ) du.x = imgu(x+1,y) - u;
    if(y < imgu.h-1 ) du.y = imgu(x,y+1) - u;
    return du;
}

//////////////////////////////////////////////////////
// Divergence operator
//////////////////////////////////////////////////////

inline __host__ __device__
float DivA(const Image<float2>& A, int x, int y)
{
    const float2 p = A(x,y);
    float divA = p.x + p.y;
    if(x>0) divA -= A(x-1,y).x;
    if(y>0) divA -= A(x,y-1).y;
    return divA;
}

//////////////////////////////////////////////////////
// TGV Epsilon operator
//////////////////////////////////////////////////////

inline __host__ __device__
float4 Epsilon(const Image<float2>& imgA, int x, int y)
{
    const float2 A = imgA(x,y);

    float dy_v0 = 0;
    float dx_v0 = 0;
    float dx_v1 = 0;
    float dy_v1 = 0;

    if (x < imgA.w-1) {
        const float2 Apx = imgA(x+1,y);
        dx_v0 = Apx.x - A.x;
        dx_v1 = Apx.y - A.y;
    }

    if (y < imgA.h-1) {
        const float2 Apy = imgA(x,y+1);
        dy_v0 = Apy.x - A.x;
        dy_v1 = Apy.y - A.y;
    }

    return make_float4(dx_v0, dy_v1, (dy_v0+dx_v1)/2.0f, (dy_v0+dx_v1)/2.0f );
}

//////////////////////////////////////////////////////
// TGV Epsilon operator transpose (generalised divergence?)
//////////////////////////////////////////////////////

inline __host__ __device__
float2 DivA(const Image<float4>& A, int x, int y)
{
    const float4 p = A(x,y);
    float2 divA = make_float2(p.x+p.z, p.z+p.y);

    if ( 0 < x ){
        divA.x -= A(x-1,y).x;
        divA.y -= A(x-1,y).z;
    }

    if ( 0 < y ){
        divA.x -= A(x,y-1).z;
        divA.y -= A(x,y-1).y;
    }

    return divA;
}

}

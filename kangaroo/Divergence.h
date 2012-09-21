#pragma once

#include "Image.h"

namespace Gpu {

inline __host__ __device__
float DivA(const Image<float2>& A, int x, int y)
{
    float divA = 0;

    const float2 p = A(x,y);

    if(x>0) divA  = p.x - A(x-1,y).x;
    if(y>0) divA += p.y - A(x,y-1).y;

    return divA;
}

}

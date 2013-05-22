#pragma once

#include "Mat.h"
#include "Image.h"

namespace roo
{

LeastSquaresSystem<float,3> PlaneFitGN(const Image<float4> dVbo, Mat<float,3,3> Qinv, Mat<float,3> zhat, Image<unsigned char> dWorkspace, Image<float> dErr, float zmin, float zmax, float c );

}

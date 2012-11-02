#include <iostream>

#include "MarchingCubes.h"
#include <kangaroo/BoundedVolume.h>
#include <kangaroo/Sdf.h>

using namespace std;


int main( int /*argc*/, char* argv[] )
{
    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetHost, Gpu::Manage> vol(16,16,16, make_float3(0,0,0), make_float3(1,1,1) );
    Gpu::SaveMesh<Gpu::SDF_t>("test",vol);
}

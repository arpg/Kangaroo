#pragma once

#include <kangaroo/BoundedVolume.h>

namespace Gpu {

template<typename T>
void SaveMesh(std::string filename, const BoundedVolume<T,TargetHost> vol );

template<typename T, typename Manage>
void SaveMesh(std::string filename, BoundedVolume<T,TargetDevice,Manage>& vol )
{
    Gpu::BoundedVolume<T,Gpu::TargetHost,Gpu::Manage> hvol(vol.w, vol.h, vol.d, vol.bbox.Min(), vol.bbox.Max());
    hvol.CopyFrom(vol);
    SaveMesh<T>(filename, hvol);
}


}

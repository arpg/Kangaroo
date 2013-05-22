#pragma once

#include "BoundedVolume.h"

namespace Gpu {

//////////////////////////////////////////
// Save SDF
//////////////////////////////////////////

template<typename T, typename TColor>
void SaveMesh(std::string filename, const BoundedVolume<T,TargetHost> vol, const BoundedVolume<TColor,TargetHost> volColor );

template<typename T, typename Manage>
void SaveMesh(std::string filename, BoundedVolume<T,TargetDevice,Manage>& vol )
{
    Gpu::BoundedVolume<T,Gpu::TargetHost,Gpu::Manage> hvol(vol.w, vol.h, vol.d, vol.bbox.Min(), vol.bbox.Max());
    Gpu::BoundedVolume<float,Gpu::TargetHost,Gpu::Manage> hvolcolor(1,1,1, vol.bbox.Min(), vol.bbox.Max() );
    hvol.CopyFrom(vol);
    SaveMesh<T,float>(filename, hvol, hvolcolor);
}

template<typename T, typename TColor, typename Manage>
void SaveMesh(std::string filename, BoundedVolume<T,TargetDevice,Manage>& vol, BoundedVolume<TColor,TargetDevice,Manage>& volColor )
{
    Gpu::BoundedVolume<T,Gpu::TargetHost,Gpu::Manage> hvol(vol.w, vol.h, vol.d, vol.bbox.Min(), vol.bbox.Max());
    Gpu::BoundedVolume<TColor,Gpu::TargetHost,Gpu::Manage> hvolcolor(volColor.w, volColor.h, volColor.d, volColor.bbox.Min(), volColor.bbox.Max());
    hvol.CopyFrom(vol);
    hvolcolor.CopyFrom(volColor);
    SaveMesh<T,TColor>(filename, hvol, hvolcolor);
}


}

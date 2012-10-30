#pragma once

#include "Image.h"

namespace Gpu
{

template<typename T, typename Target = TargetDevice, typename Management = DontManage>
struct ImageKeyframe
{
    Gpu::Mat<float,3,4> KiT_iw;
    Gpu::Image<T, Gpu::TargetDevice> img;
};

} // namespace Gpu

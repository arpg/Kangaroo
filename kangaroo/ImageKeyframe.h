#pragma once

#include "Image.h"
#include "MatUtils.h"
#include "ImageIntrinsics.h"

namespace Gpu
{

template<typename T, typename Target = TargetDevice, typename Management = DontManage>
struct ImageKeyframe : public ImageTransformProject
{
    Image<T, Gpu::TargetDevice> img;
};

} // namespace Gpu

#pragma once

#include "Image.h"
#include "MatUtils.h"
#include "ImageIntrinsics.h"

namespace roo
{

template<typename T, typename Target = TargetDevice, typename Management = DontManage>
struct ImageKeyframe : public ImageTransformProject
{
    Image<T, roo::TargetDevice> img;
};

} // namespace roo

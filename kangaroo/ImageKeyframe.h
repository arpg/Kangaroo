#pragma once

#include <kangaroo/Image.h>
#include "MatUtils.h"
#include <kangaroo/ImageIntrinsics.h>

namespace roo
{

template<typename T, typename Target = TargetDevice, typename Management = DontManage>
struct ImageKeyframe : public ImageTransformProject
{
    Image<T, roo::TargetDevice> img;
};

} // namespace roo

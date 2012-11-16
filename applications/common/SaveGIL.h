#pragma once

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#include <kangaroo/Image.h>

void SaveGIL(const std::string filename, const Gpu::Image<uchar3,Gpu::TargetHost>& image)
{
    boost::gil::rgb8_view_t v = boost::gil::interleaved_view(image.w, image.h, reinterpret_cast<boost::gil::rgb8_ptr_t>(image.ptr), image.pitch);
    boost::gil::png_write_view(filename, v );
}

template<typename T, typename Manage>
void SaveGIL(const std::string filename, const Gpu::Image<T,Gpu::TargetDevice,Manage>& img)
{
    Gpu::Image<T,Gpu::TargetHost,Gpu::Manage> himg(img.w, img.h);
    himg.CopyFrom(img);
    SaveGIL(filename, himg);
}

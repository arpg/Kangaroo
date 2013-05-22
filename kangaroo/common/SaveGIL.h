#pragma once

#include <kangaroo/Image.h>

#include <boost/gil/gil_all.hpp>

// TODO Test for PNG precence
#define HAVE_PNG

#ifdef HAVE_PNG
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>
#endif // HAVE_PNG

void SaveGIL(const std::string filename, const Gpu::Image<uchar3,Gpu::TargetHost>& image)
{
    boost::gil::rgb8_view_t v = boost::gil::interleaved_view(image.w, image.h, reinterpret_cast<boost::gil::rgb8_ptr_t>(image.ptr), image.pitch);
    
#ifdef HAVE_PNG
    boost::gil::png_write_view(filename, v );
#else
    std::cerr << "libpng support not compiled in" << std::endl;
#endif
}

template<typename T, typename Manage>
void SaveGIL(const std::string filename, const Gpu::Image<T,Gpu::TargetDevice,Manage>& img)
{
    Gpu::Image<T,Gpu::TargetHost,Gpu::Manage> himg(img.w, img.h);
    himg.CopyFrom(img);
    SaveGIL(filename, himg);
}

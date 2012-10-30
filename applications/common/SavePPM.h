#pragma once

#include <kangaroo/Image.h>
#include <kangaroo/Volume.h>

#include <iostream>

// P1	Portable bitmap	ASCII
// P2	Portable graymap	ASCII
// P3	Portable pixmap	ASCII
// P4	Portable bitmap	Binary
// P5	Portable graymap	Binary
// P6	Portable pixmap	Binary

/////////////////////////////////////////////////////////////////////////////
// Image types
/////////////////////////////////////////////////////////////////////////////

template<typename T>
void SavePXM(const std::string filename, const Gpu::Image<T,Gpu::TargetHost>& image, std::string ppm_type = "P5", int num_colors = 255)
{
    std::ofstream bFile( filename.c_str(), std::ios::out | std::ios::binary );
    bFile << ppm_type << std::endl;
    bFile << image.w << " " << image.h << std::endl;
    bFile << num_colors << std::endl;
    for(int r=0; r<image.h; ++r) {
        bFile.write( (const char*)image.RowPtr(r), image.w * sizeof(T) );
    }
    bFile.close();
}

template<typename T>
void SavePXM(const std::string filename, const Gpu::Image<T,Gpu::TargetDevice>& image, std::string ppm_type = "P5", int num_colors = 255)
{
    Gpu::Image<T,Gpu::TargetHost> himage(image.w, image.h);
    himage.CopyFrom(image);
    SavePXM(filename, himage, ppm_type, num_colors);
}

/////////////////////////////////////////////////////////////////////////////
// Volume types
/////////////////////////////////////////////////////////////////////////////

template<typename T>
void SavePXM(const std::string filename, const Gpu::Volume<T,Gpu::TargetHost>& vol, std::string ppm_type = "P5", int num_colors = 255)
{
    std::ofstream bFile( filename.c_str(), std::ios::out | std::ios::binary );
    bFile << ppm_type << std::endl;
    bFile << vol.w << " " << vol.h << " " << vol.d << std::endl;
    bFile << num_colors << std::endl;
    for(int d=0; d<vol.d; ++d) {
        for(int r=0; r<vol.h; ++r) {
            bFile.write( (const char*)vol.RowPtr(r,d), vol.w * sizeof(T) );
        }
    }
    bFile.close();
}

template<typename T>
void SavePXM(const std::string filename, const Gpu::Volume<T,Gpu::TargetDevice>& vol, std::string ppm_type = "P5", int num_colors = 255)
{
    Gpu::Volume<T,Gpu::TargetHost> hvol(vol.w, vol.h, vol.d);
    hvol.CopyFrom(vol);
    SavePXM(filename, hvol, ppm_type, num_colors);
}

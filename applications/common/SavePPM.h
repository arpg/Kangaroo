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
    bFile << image.w << " " << image.h << '\n';
    bFile << num_colors << '\n';
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
// Save Volume types
/////////////////////////////////////////////////////////////////////////////

template<typename T, typename Manage>
void SavePXM(const std::string filename, const Gpu::Volume<T,Gpu::TargetHost,Manage>& vol, std::string ppm_type = "P5", int num_colors = 255)
{
    std::ofstream bFile( filename.c_str(), std::ios::out | std::ios::binary );
    bFile << ppm_type << std::endl;
    bFile << vol.w << " " << vol.h << " " << vol.d << '\n';
    bFile << num_colors << '\n';
    for(int d=0; d<vol.d; ++d) {
        for(int r=0; r<vol.h; ++r) {
            bFile.write( (const char*)vol.RowPtr(r,d), vol.w * sizeof(T) );
        }
    }
    bFile.close();
}

template<typename T, typename Manage>
void SavePXM(const std::string filename, const Gpu::Volume<T,Gpu::TargetDevice,Manage>& vol, std::string ppm_type = "P5", int num_colors = 255)
{
    Gpu::Volume<T,Gpu::TargetHost,Gpu::Manage> hvol(vol.w, vol.h, vol.d);
    hvol.CopyFrom(vol);
    SavePXM(filename, hvol, ppm_type, num_colors);
}

/////////////////////////////////////////////////////////////////////////////
// Load Volume types
/////////////////////////////////////////////////////////////////////////////

template<typename T>
bool LoadPXM(const std::string filename, Gpu::Volume<T,Gpu::TargetHost,Gpu::Manage>& vol)
{
    // Parse header
    std::ifstream bFile( filename.c_str(), std::ios::in | std::ios::binary );
    std::string ppm_type = "";
    int num_colors = 0;
    int w = 0;
    int h = 0;
    int d = 0;

    bFile >> ppm_type;
    bFile >> w;
    bFile >> h;
    bFile >> d;
    bFile >> num_colors;
    bFile.ignore(1,'\n');

    bool success = !bFile.fail() && w > 0 && h > 0 && d > 0;

    if(success) {
        // Make sure vol is empty
        Gpu::Manage::Cleanup<T,Gpu::TargetHost>(vol.ptr);

        // Allocate memory
        Gpu::TargetHost::AllocatePitchedMem<T>(&vol.ptr,&vol.pitch,&vol.img_pitch,w,h,d);
        vol.w = w; vol.h = h; vol.d = d;

        // Read in data
        for(int d=0; d<vol.d; ++d) {
            for(int r=0; r<vol.h; ++r) {
                bFile.read( (char*)vol.RowPtr(r,d), vol.w * sizeof(T) );
            }
        }
        success = !bFile.fail();
    }
    bFile.close();

    return success;
}

template<typename T>
bool LoadPXM(const std::string filename, Gpu::Volume<T,Gpu::TargetDevice,Gpu::Manage>& vol)
{
    Gpu::Volume<T,Gpu::TargetHost,Gpu::Manage> hvol;
    bool success = LoadPXM(filename, hvol);

    if(success) {
        Gpu::Manage::Cleanup<T,Gpu::TargetDevice>(vol.ptr);

        Gpu::TargetDevice::AllocatePitchedMem<T>(&vol.ptr,&vol.pitch,&vol.img_pitch,hvol.w,hvol.h,hvol.d);
        vol.w = hvol.w; vol.h = hvol.h; vol.d = hvol.d;

        vol.CopyFrom(hvol);
    }
    return success;
}

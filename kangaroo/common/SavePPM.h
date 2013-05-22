#pragma once

#include <kangaroo/Image.h>
#include <kangaroo/Volume.h>
#include <Kangaroo/BoundedVolume.h>

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
void SavePXM(const std::string filename, const roo::Image<T,roo::TargetHost>& image, std::string ppm_type = "P5", int num_colors = 255)
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
void SavePXM(const std::string filename, const roo::Image<T,roo::TargetDevice>& image, std::string ppm_type = "P5", int num_colors = 255)
{
    roo::Image<T,roo::TargetHost> himage(image.w, image.h);
    himage.CopyFrom(image);
    SavePXM(filename, himage, ppm_type, num_colors);
}

/////////////////////////////////////////////////////////////////////////////
// Save Volume types
/////////////////////////////////////////////////////////////////////////////

template<typename T, typename Manage>
void SavePXM(std::ofstream& bFile, const roo::Volume<T,roo::TargetHost,Manage>& vol, std::string ppm_type = "P5", int num_colors = 255)
{
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
void SavePXM(const std::string filename, const roo::Volume<T,roo::TargetHost,Manage>& vol, std::string ppm_type = "P5", int num_colors = 255)
{
    std::ofstream bFile( filename.c_str(), std::ios::out | std::ios::binary );
    SavePXM<T,Manage>(bFile, vol, ppm_type, num_colors);
}

template<typename T, typename Manage>
void SavePXM(std::ofstream& bFile, const roo::Volume<T,roo::TargetDevice,Manage>& vol, std::string ppm_type = "P5", int num_colors = 255)
{
    roo::Volume<T,roo::TargetHost,roo::Manage> hvol(vol.w, vol.h, vol.d);
    hvol.CopyFrom(vol);
    SavePXM(bFile, hvol, ppm_type, num_colors);
}



template<typename T, typename Manage>
void SavePXM(const std::string filename, const roo::BoundedVolume<T,roo::TargetDevice,Manage>& vol, std::string ppm_type = "P5", int num_colors = 255)
{
    std::ofstream bFile( filename.c_str(), std::ios::out | std::ios::binary );

    bFile << vol.bbox.boxmin.x << " " <<  vol.bbox.boxmin.y << " " << vol.bbox.boxmin.z << std::endl;
    bFile << vol.bbox.boxmax.x << " " <<  vol.bbox.boxmax.y << " " << vol.bbox.boxmax.z << std::endl;
    SavePXM<T,Manage>(bFile,vol,ppm_type,num_colors);
}

template<typename T, typename Manage>
void SavePXM(const std::string filename, const roo::Volume<T,roo::TargetDevice,Manage>& vol, std::string ppm_type = "P5", int num_colors = 255)
{
    std::ofstream bFile( filename.c_str(), std::ios::out | std::ios::binary );
    SavePXM<T,Manage>(bFile,vol,ppm_type,num_colors);
}


/////////////////////////////////////////////////////////////////////////////
// Load Volume types
/////////////////////////////////////////////////////////////////////////////

template<typename T>
bool LoadPXM(std::ifstream& bFile, roo::Volume<T,roo::TargetHost,roo::Manage>& vol)
{
    // Parse header

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
        roo::Manage::Cleanup<T,roo::TargetHost>(vol.ptr);

        // Allocate memory
        roo::TargetHost::AllocatePitchedMem<T>(&vol.ptr,&vol.pitch,&vol.img_pitch,w,h,d);
        vol.w = w; vol.h = h; vol.d = d;

        // Read in data
        for(size_t d=0; d<vol.d; ++d) {
            for(size_t r=0; r<vol.h; ++r) {
                bFile.read( (char*)vol.RowPtr(r,d), vol.w * sizeof(T) );
            }
        }
        success = !bFile.fail();
    }
    bFile.close();

    return success;
}

template<typename T>
bool LoadPXM(const std::string filename, roo::Volume<T,roo::TargetHost,roo::Manage>& vol)
{
    std::ifstream bFile( filename.c_str(), std::ios::in | std::ios::binary );
    return LoadPXM<T>(bFile,vol);
}

template<typename T>
bool LoadPXM(const std::string filename, roo::BoundedVolume<T,roo::TargetHost,roo::Manage>& vol)
{
    std::ifstream bFile( filename.c_str(), std::ios::in | std::ios::binary );
    //read in the bounding volume bounds
    bFile >> vol.bbox.boxmin.x;
    bFile >> vol.bbox.boxmin.y;
    bFile >> vol.bbox.boxmin.z;
    bFile >> vol.bbox.boxmax.x;
    bFile >> vol.bbox.boxmax.y;
    bFile >> vol.bbox.boxmax.z;
    bFile.ignore(1,'\n');
    return LoadPXM<T>(bFile,vol);
}


template<typename T>
bool LoadPXM(const std::string filename, roo::Volume<T,roo::TargetDevice,roo::Manage>& vol)
{
    roo::Volume<T,roo::TargetHost,roo::Manage> hvol;
    bool success = LoadPXM(filename, hvol);

    if(success) {
        roo::Manage::Cleanup<T,roo::TargetDevice>(vol.ptr);

        roo::TargetDevice::AllocatePitchedMem<T>(&vol.ptr,&vol.pitch,&vol.img_pitch,hvol.w,hvol.h,hvol.d);
        vol.w = hvol.w; vol.h = hvol.h; vol.d = hvol.d;

        vol.CopyFrom(hvol);
    }
    return success;
}

template<typename T>
bool LoadPXM(const std::string filename, roo::BoundedVolume<T,roo::TargetDevice,roo::Manage>& vol)
{
    roo::BoundedVolume<T,roo::TargetHost,roo::Manage> hvol;
    bool success = LoadPXM(filename, hvol);

    if(success) {
        roo::Manage::Cleanup<T,roo::TargetDevice>(vol.ptr);

        roo::TargetDevice::AllocatePitchedMem<T>(&vol.ptr,&vol.pitch,&vol.img_pitch,hvol.w,hvol.h,hvol.d);
        vol.w = hvol.w; vol.h = hvol.h; vol.d = hvol.d;

        vol.CopyFrom(hvol);
        vol.bbox = hvol.bbox;
    }
    return success;
}

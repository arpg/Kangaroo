#pragma once

#include <kangaroo/Image.h>
#include <iostream>

// P1	Portable bitmap	ASCII
// P2	Portable graymap	ASCII
// P3	Portable pixmap	ASCII
// P4	Portable bitmap	Binary
// P5	Portable graymap	Binary
// P6	Portable pixmap	Binary

template<typename T>
void SavePXM(std::string filename, Gpu::Image<T,Gpu::TargetHost> image, std::string ppm_type = "P5", int num_colors = 255)
{
    std::ofstream bFile( filename.c_str(), std::ios::out | std::ios::binary );
    bFile << ppm_type << std::endl;
    bFile << image.w << " " << image.h << std::endl;
    bFile << num_colors << std::endl;
    for(int r=0; r<image.h; ++r) {
        bFile.write( (const char*)image.RowPtr(r), image.w );
    }
    bFile.close();
}

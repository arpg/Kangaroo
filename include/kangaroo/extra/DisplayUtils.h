#pragma once

#include <pangolin/glcuda.h>
#include <kangaroo/Image.h>

inline void ApplyPreferredGlSettings()
{
    // Disable multisample for general use (it messes up antialiased lines)
    // Enable individually for particular GLObjects
    glDisable(GL_MULTISAMPLE);
    
//    // GL_POLYGON_SMOOTH is known to be really bad.
//    glEnable(GL_POLYGON_SMOOTH);
//    glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST );
    
    glEnable(GL_NORMALIZE);

    // Antialiased lines work great, but should be disabled if multisampling is enabled
    glEnable(GL_LINE_SMOOTH);
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
    glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );

    // Enable alpha blending
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Shading model to use when lighing is enabled
    glShadeModel(GL_SMOOTH);
    
    glDepthFunc( GL_LEQUAL );
    glEnable( GL_DEPTH_TEST );

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    glLineWidth(1.5);
}

template<typename T, typename Owner>
inline void CopyToTex(pangolin::GlTextureCudaArray& tex, const roo::Image<T,roo::TargetDevice,Owner>& dImg)
{
    pangolin::CudaScopedMappedArray arr_tex(tex);
    cudaError_t err = cudaMemcpy2DToArray(*arr_tex, 0, 0, dImg.ptr, dImg.pitch, sizeof(T)*min(tex.width,dImg.w), min(tex.height,dImg.h), cudaMemcpyDeviceToDevice );
    if( err != cudaSuccess ) {
        std::cerr << "cudaMemcpy2DToArray failed: " << err << std::endl;
    }
}

template<typename T, typename Owner>
inline void operator<<(pangolin::GlTextureCudaArray& tex, const roo::Image<T,roo::TargetDevice,Owner>& dImg)
{
    CopyToTex(tex,dImg);
}

#pragma once

#include <pangolin/glcuda.h>

#include "../cu/Image.h"

template<typename T, typename Owner>
inline void operator<<(pangolin::GlTextureCudaArray& tex, const Gpu::Image<T,Gpu::TargetDevice,Owner>& dImg)
{
    pangolin::CudaScopedMappedArray arr_tex(tex);
    cudaError_t err = cudaMemcpy2DToArray(*arr_tex, 0, 0, dImg.ptr, dImg.pitch, dImg.w*sizeof(T), dImg.h, cudaMemcpyDeviceToDevice );
    if( err != cudaSuccess ) {
        std::cerr << "cudaMemcpy2DToArray failed: " << err << std::endl;
    }
}

inline void RenderVbo(pangolin::GlBufferCudaPtr& vbo, int w, int h)
{
    vbo.Bind();
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glPointSize(2.0);
    glDrawArrays(GL_POINTS, 0, w * h);
    glDisableClientState(GL_VERTEX_ARRAY);
    vbo.Unbind();
}

inline void RenderVboCbo(pangolin::GlBufferCudaPtr& vbo, pangolin::GlBufferCudaPtr& cbo, int w, int h, bool draw_color = true)
{
    if(draw_color) {
        cbo.Bind();
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
        glEnableClientState(GL_COLOR_ARRAY);
    }

    vbo.Bind();
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glPointSize(2.0);
    glDrawArrays(GL_POINTS, 0, w * h);

    if(draw_color) {
        glDisableClientState(GL_COLOR_ARRAY);
        cbo.Unbind();
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    vbo.Unbind();
}

inline void RenderVboIbo(pangolin::GlBufferCudaPtr& vbo, pangolin::GlBufferCudaPtr& ibo, int w, int h, bool draw_mesh = true)
{
    vbo.Bind();
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    if(draw_mesh) {
        ibo.Bind();
        for( int r=0; r<h-1; ++r) {
            glDrawElements(GL_TRIANGLE_STRIP,2*w, GL_UNSIGNED_INT, (unsigned int*)0 + 2*w*r);
        }
        ibo.Unbind();
    }else{
        glPointSize(2.0);
        glDrawArrays(GL_POINTS, 0, w * h);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    vbo.Unbind();
}

inline void RenderVboIboCbo(pangolin::GlBufferCudaPtr& vbo, pangolin::GlBufferCudaPtr& ibo, pangolin::GlBufferCudaPtr& cbo, int w, int h, bool draw_mesh = true, bool draw_color = true)
{
    if(draw_color) {
        cbo.Bind();
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
        glEnableClientState(GL_COLOR_ARRAY);
    }

    vbo.Bind();
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    if(draw_mesh) {
        ibo.Bind();
        for( int r=0; r<h-1; ++r) {
            glDrawElements(GL_TRIANGLE_STRIP,2*w, GL_UNSIGNED_INT, (unsigned int*)0 + 2*w*r);
        }
        ibo.Unbind();
    }else{
        glPointSize(2.0);
        glDrawArrays(GL_POINTS, 0, w * h);
    }

    if(draw_color) {
        glDisableClientState(GL_COLOR_ARRAY);
        cbo.Unbind();
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    vbo.Unbind();
}

inline void RenderVboIboCboNbo(pangolin::GlBufferCudaPtr& vbo, pangolin::GlBufferCudaPtr& ibo, pangolin::GlBufferCudaPtr& cbo, pangolin::GlBufferCudaPtr& nbo, int w, int h, bool draw_mesh = true, bool draw_color = true, bool draw_normals = true)
{
    if(draw_color) {
        cbo.Bind();
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
        glEnableClientState(GL_COLOR_ARRAY);
    }

    vbo.Bind();
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    if(draw_mesh) {
        if(draw_normals) {
            nbo.Bind();
            glNormalPointer(GL_FLOAT,sizeof(float4),0);
            glEnableClientState(GL_NORMAL_ARRAY);
        }

        ibo.Bind();
        for( int r=0; r<h-1; ++r) {
            glDrawElements(GL_TRIANGLE_STRIP,2*w, GL_UNSIGNED_INT, (unsigned int*)0 + 2*w*r);
        }
        ibo.Unbind();

        if(draw_normals) {
            glDisableClientState(GL_NORMAL_ARRAY);
            nbo.Unbind();
        }
    }else{
        glPointSize(2.0);
        glDrawArrays(GL_POINTS, 0, w * h);
    }

    if(draw_color) {
        glDisableClientState(GL_COLOR_ARRAY);
        cbo.Unbind();
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    vbo.Unbind();
}

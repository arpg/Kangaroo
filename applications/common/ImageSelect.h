#pragma once

#include <Pangolin/pangolin.h>
#include <Pangolin/gl.h>
#include <Pangolin/glsl.h>

#include "DisplayUtils.h"

#include <kangaroo/Image.h>
#include <kangaroo/Pyramid.h>

#include <algorithm>

#ifdef HAVE_EIGEN
#include <Eigen/Eigen>
#endif // HAVE_EIGEN

namespace pangolin
{

class ImageSelect
{
public:
    ImageSelect(int w, int h, int level = 0)
        : img_w(w), img_h(h), level(level), selected(false), pixel_scale(1.0)
    {
        topleft[0] = 0;
        topleft[1] = 0;
    }

#ifdef HAVE_EIGEN
    Eigen::Vector2d GetSelectedPoint(bool flipy = false) const
    {
        return Eigen::Vector2d(topleft[0], flipy ? (img_h-1) - topleft[1] : topleft[1]);
    }
#endif

    bool IsSelected() const {
        return selected;
    }

    void WindowToImage(const Viewport& v, int wx, int wy, float& ix, float& iy )
    {
        ix = img_w * (wx - v.l) /(float)v.w - 0.5;
        iy = img_h * (wy - v.b) /(float)v.h - 0.5;
        ix = std::max(0.0f,std::min(ix, img_w-1.0f));
        iy = std::max(0.0f,std::min(iy, img_h-1.0f));
    }

    void ImageToWindow(const Viewport& v, float ix, float iy, float& wx, float& wy )
    {
        wx = v.l + (float)v.w * ix / img_w;
        wy = v.b + (float)v.h * iy / img_h;
    }

    float PixelScale()
    {
        return pixel_scale;
    }

    void SetPixelScale(float scale)
    {
        pixel_scale = scale;
    }

    int Level()
    {
        return level;
    }

protected:
    float img_w, img_h;
    int level;
    bool selected;
    float topleft[2];
    float pixel_scale;
};

class Handler2dImageSelect : public Handler, public ImageSelect
{
public:
    Handler2dImageSelect(int w, int h, int level=0)
        : ImageSelect(w,h,level)
    {
    }

    virtual void Keyboard(View&, unsigned char key, int x, int y, bool pressed)
    {
        if(key == 'r') {
            selected = false;
            pixel_scale = 1.0;
        }
    }

    virtual void Mouse(View& view, MouseButton button, int x, int y, bool pressed, int button_state)
    {
        if(button == MouseWheelUp) {
            pixel_scale *= 1.02;
        }else if(button == MouseWheelDown) {
            pixel_scale *= 0.98;
        }else{
            WindowToImage(view.v, x,y, topleft[0], topleft[1]);
            selected = (button == pangolin::MouseButtonLeft);
        }
    }

    virtual void MouseMotion(View& view, int x, int y, int button_state)
    {
        WindowToImage(view.v, x,y, topleft[0], topleft[1]);
    }
};

void RenderImageSelect(const ImageSelect& is, int imgw, int imgh, int inputlevel = 0)
{
    if(is.IsSelected()) {
        glColor3f(1,0,0);
        Eigen::Vector2d p = is.GetSelectedPoint();
        p[0] = (1<<inputlevel) * (p[0]+0.5) * 2.0 / imgw - 1;
        p[1] = (1<<inputlevel) * (p[1]+0.5) * 2.0 / imgh - 1;
        glDrawCross(p);
        glColor3f(1,1,1);
    }
}

void RenderToViewport(const GlTexture& glTex, bool flipy, float pixScale)
{
    if(pixScale!=1.0) {
        GlSlUtilities::Scale(pixScale);
        glTex.RenderToViewport(flipy);
        GlSlUtilities::UseNone();
    }else{
        glTex.RenderToViewport(flipy);
    }
}

class ActivateDrawTexture
{
public:
    ActivateDrawTexture(const GlTexture& glTex, bool flipy=false)
        :glTex(glTex), flipy(flipy)
    {
    }

    void operator()(pangolin::View& view) {
        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        view.Activate();
        glColor3f(1,1,1);

        ImageSelect* imageSelect = dynamic_cast<ImageSelect*>(view.handler);

        if(imageSelect) {
            const float pixScale = imageSelect->PixelScale();
            RenderToViewport(glTex,flipy,pixScale);
            RenderImageSelect(*imageSelect, glTex.width, glTex.height);
        }else{
            glTex.RenderToViewport(flipy);
        }

        glPopAttrib();
    }

protected:
    const GlTexture& glTex;
    bool flipy;
};

template<typename T>
class ActivateDrawImage
{
public:
    ActivateDrawImage(Gpu::Image<T,Gpu::TargetDevice> image, GLint internal_format = GL_RGBA8, bool sampling_linear = true, bool flipy=false)
        :image(image), glTex(image.w,image.h,internal_format,sampling_linear), flipy(flipy), pixScale(1)
    {
    }

    void operator()(pangolin::View& view) {
        CopyToTex<T,Gpu::DontManage>(glTex,image);

        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        view.Activate();
        glColor3f(1,1,1);

        ImageSelect* imageSelect = dynamic_cast<ImageSelect*>(view.handler);

        if(imageSelect) {
            const float doScale = pixScale * imageSelect->PixelScale();
            RenderToViewport(glTex,flipy,doScale);
            RenderImageSelect(*imageSelect, glTex.width, glTex.height);
        }else{
            RenderToViewport(glTex,flipy,pixScale);
        }

        glPopAttrib();
    }

    void SetImage(Gpu::Image<T,Gpu::TargetDevice> image) {
        this->image.ptr = image.ptr;
        this->image.pitch = image.pitch;
        this->image.w = image.w;
        this->image.h = image.h;
    }

    void SetImageScale(float scale) {
        pixScale = scale;
    }

protected:    
    Gpu::Image<T,Gpu::TargetDevice> image;
    pangolin::GlTextureCudaArray glTex;
    bool flipy;
    float pixScale;

private:
    // Private copy constructor
    ActivateDrawImage(const ActivateDrawImage<T>& adi) {}
};

template<typename T, unsigned Levels>
class ActivateDrawPyramid
{
public:
    ActivateDrawPyramid(const Gpu::Pyramid<T,Levels> pyramid, GLint internal_format = GL_RGBA8, bool sampling_linear = true, bool flipy=false)
        :pyramid(pyramid), glTex(pyramid[0].w,pyramid[0].h,internal_format,sampling_linear), flipy(flipy), level(0)
    {
    }

    void operator()(pangolin::View& view) {
        CopyToTex<T,Gpu::DontManage>(glTex,pyramid[level]);

        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        view.Activate();
        glColor3f(1,1,1);

        ImageSelect* imageSelect = dynamic_cast<ImageSelect*>(view.handler);

        glMatrixMode(GL_TEXTURE);
        glScalef(1.0f/(1<<level), 1.0f/(1<<level),1.0f);

        if(imageSelect) {
            const float pixScale = imageSelect->PixelScale();
            RenderToViewport(glTex,flipy,pixScale);
            RenderImageSelect(*imageSelect, glTex.width, glTex.height, imageSelect->Level());
        }else{
            glTex.RenderToViewport(flipy);
        }

        glMatrixMode(GL_TEXTURE);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);

        glPopAttrib();
    }

    void SetLevel(unsigned l)
    {
        level = l;
    }

protected:
    const Gpu::Pyramid<T,Levels> pyramid;
    pangolin::GlTextureCudaArray glTex;
    bool flipy;
    unsigned level;

private:
    // Private copy constructor
    ActivateDrawPyramid(const ActivateDrawPyramid<T,Levels>& adi) {}
};

}

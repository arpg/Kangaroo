#pragma once

#include <Pangolin/pangolin.h>
#include <Pangolin/gl.h>
#include <Pangolin/glsl.h>

#include <algorithm>

#ifdef HAVE_EIGEN
#include <Eigen/Eigen>
#endif // HAVE_EIGEN

namespace pangolin
{

class ImageSelect
{
public:
    ImageSelect(int w, int h)
        : img_w(w), img_h(h), selected(false), pixel_scale(1.0)
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

protected:
    float img_w, img_h;
    bool selected;
    float topleft[2];
    float pixel_scale;
};

class Handler2dImageSelect : public Handler, public ImageSelect
{
public:
    Handler2dImageSelect(int w, int h)
        : ImageSelect(w,h)
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

        ImageSelect* imageSelect = dynamic_cast<ImageSelect*>(view.handler);

        if(imageSelect) {
            float pixScale = imageSelect->PixelScale();
            if(pixScale!=1.0) {
                GlSlUtilities::Scale(pixScale);
                glTex.RenderToViewport(flipy);
                GlSlUtilities::UseNone();
            }else{
                glTex.RenderToViewport(flipy);
            }

            if(imageSelect->IsSelected()) {
                glColor3f(1,0,0);
                Eigen::Vector2d p = imageSelect->GetSelectedPoint();
                p[0] = (p[0]+0.5) * 2.0 / glTex.width - 1;
                p[1] = (p[1]+0.5) * 2.0 / glTex.height - 1;
                glDrawCross(p);
                glColor3f(1,1,1);
            }
        }else{
            glTex.RenderToViewport(flipy);
        }
        glPopAttrib();
    }

protected:
    const GlTexture& glTex;
    bool flipy;
};

}

#include <pangolin/pangolin.h>
#include <pangolin/simple_math.h>
#include <kangaroo/Image.h>

struct Handler3DGpuDepth : public pangolin::Handler3D
{
    Handler3DGpuDepth(Gpu::Image<float> depth, pangolin::OpenGlRenderState& cam_state, pangolin::AxisDirection enforce_up=pangolin::AxisNone, float trans_scale=0.01f)
        : pangolin::Handler3D(cam_state,enforce_up, trans_scale), depth(depth)
    {
    }

    void GetPosNormal(pangolin::View& view, int x, int y, double p[3], double Pw[3], double Pc[3], double n[3], double default_z)
    {
        const GLint viewport[4] = {view.v.l,view.v.b,view.v.w,view.v.h};
        const pangolin::OpenGlMatrix proj = cam_state->GetProjectionMatrix();
        const pangolin::OpenGlMatrix mv = cam_state->GetModelViewMatrix();

        float z = 0;
        const int imx = depth.w * (float)(x-view.v.l) / (float)view.v.w;
        const int imy = depth.h * (float)(view.v.t()-y) / (float)view.v.h;
        depth.SubImage(imx, imy, 1,1).MemcpyToHost(&z);
        const float zw = 0.5*(1 + proj.m[2*4+2] + proj.m[3*4+2] / z);
        gluUnProject(x, y, zw, mv.m, proj.m, viewport, &Pw[0], &Pw[1], &Pw[2]);
        pangolin::LieApplySE34x4vec3(Pc, mv.m, Pw);
        p[0] = x; p[1] = y; p[2] = std::isfinite(z) ? zw : default_z;
    }

protected:
    Gpu::Image<float> depth;
};

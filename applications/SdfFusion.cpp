#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <npp.h>

#include <SceneGraph/SceneGraph.h>

#include "common/RpgCameraOpen.h"
#include "common/DisplayUtils.h"
#include "common/BaseDisplayCuda.h"
#include "common/ImageSelect.h"
#include "common/Handler3dGpuDepth.h"
#include "common/CameraModelPyramid.h"

#include <kangaroo/kangaroo.h>

using namespace std;
using namespace pangolin;

int main( int argc, char* argv[] )
{
    const unsigned int w = 512;
    const unsigned int h = 512;
    const float u0 = w /2;
    const float v0 = h /2;
    const float fu = 500;
    const float fv = 500;

    // Initialise window
    View& container = SetupPangoGLWithCuda(2*w, h);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

    Var<float> near("ui.near",0, 0, 10);
    Var<float> far("ui.far",100, 0, 100);

    // Allocate Camera Images on device for processing
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> img(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> depth(w,h);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> norm(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> gtd(w,h);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> gtn(w,h);

    pangolin::GlBufferCudaPtr vbo(pangolin::GlArrayBuffer, w,h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);

//    Gpu::Volume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(8,8,8);
    Gpu::Volume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(64,64,64);
//    Gpu::Volume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(256,256,256);
    ActivateDrawImage<float> adg(img, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawImage<float4> adn(norm, GL_RGBA32F, true, true);

    SceneGraph::GLSceneGraph graph;
    SceneGraph::GLAxis glaxis;
    SceneGraph::GLAxisAlignedBox glbox;
    SceneGraph::GLVbo glvbo(&vbo);
    graph.AddChild(&glaxis);
    graph.AddChild(&glbox);
    graph.AddChild(&glvbo);

    pangolin::OpenGlRenderState stacks_view(
        ProjectionMatrixRDF_TopLeft(w,h, fu,fv, u0,v0, 1E-2,1E3),
        ModelViewLookAtRDF(0,0,-4,0,0,1,0,-1,0)
    );

    pangolin::OpenGlRenderState stacks_capture(
        ProjectionMatrixRDF_TopLeft(w,h, fu,fv, u0,v0, 1E-2,1E3),
        ModelViewLookAtRDF(0,0,-4,0,0,1,0,-1,0)
    );

    Handler3DGpuDepth handler(depth,stacks_view, AxisNone);
    SceneGraph::HandlerSceneGraph handlerView(graph, stacks_view, AxisNone);
    SceneGraph::HandlerSceneGraph handlerCapture(graph, stacks_capture, AxisNone);
    SetupContainer(container, 4, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adg)).SetHandler(&handler);
    container[1].SetDrawFunction(boost::ref(adn)).SetHandler(&handler);
    container[2].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, stacks_view)).SetHandler( &handlerView  );
    container[3].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, stacks_capture)).SetHandler( &handlerCapture  );

    const float3 boxmax = make_float3(1,1,1);
    const float3 boxmin = make_float3(-1,-1,-1);
    Gpu::SdfSphere(vol, boxmin, boxmax, make_float3(0,0,0), 0.9 );

    Var<bool> subpix("ui.subpix", true, true);
    Var<bool> sdfreset("ui.reset", false, false);
    Var<bool> sdfsphere("ui.sphere", false, false);
    Var<bool> fuse("ui.fuse", false, true);
    Var<bool> fuseonce("ui.fuse once", false, false);
    Var<float> trunc_dist("ui.trunc dist", 0.3, 0,0.5);
    Var<float> max_w("ui.max w", 10, 1E-4, 10);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        Sophus::SE3 T_vw(stacks_view.GetModelViewMatrix());
        Sophus::SE3 T_cw(stacks_capture.GetModelViewMatrix());

        if(Pushed(sdfreset)) {
            Gpu::SdfReset(vol, trunc_dist);
        }

        if(Pushed(sdfsphere)) {
            Gpu::SdfSphere(vol, boxmin, boxmax, make_float3(0,0,0), 0.9 );
        }

        // Raycast current view
        {
            Gpu::Raycast(depth, norm, img, vol, boxmin, boxmax, T_vw.inverse().matrix3x4(), fu, fv, u0, v0, near, far, trunc_dist, subpix );
        }

        // Generate depthmap by raycasting against groundtruth object
        {
            Gpu::RaycastSphere(gtd, T_cw.inverse().matrix3x4(), fu, fv, u0, v0, make_float3(0,0,0), 0.9);
            CudaScopedMappedPtr dvbo(vbo);
            Gpu::Image<float4> vboimg((float4*)*dvbo,w,h);
            Gpu::DepthToVbo(vboimg, gtd, fu,fv,u0,v0, 1.0f);
            Gpu::NormalsFromVbo(gtn,vboimg);
            glvbo.SetPose(T_cw.inverse().matrix());
        }

        if(Pushed(fuseonce) || fuse) {
            // integrate gtd into TSDF
            Gpu::SdfFuse(vol, boxmin, boxmax, gtd, gtn, T_cw.matrix3x4(), fu, fv, u0, v0, trunc_dist, max_w );
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

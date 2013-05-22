#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include <SceneGraph/SceneGraph.h>

#include <sophus/se3.hpp>

#include <kangaroo/kangaroo.h>
#include <kangaroo/common/DisplayUtils.h>
#include <kangaroo/common/BaseDisplayCuda.h>
#include <kangaroo/common/ImageSelect.h>
#include <kangaroo/common/Handler3dGpuDepth.h>

using namespace std;
using namespace pangolin;

int main( int argc, char* argv[] )
{
    const unsigned int w = 512;
    const unsigned int h = 512;
    const roo::ImageIntrinsics K(500,500,w /2, h /2);
    const int volres = 128;

    // Initialise window
    View& container = SetupPangoGLWithCuda(2*w, h);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

    Var<float> near("ui.near",0, 0, 10);
    Var<float> far("ui.far",10, 0, 10);

    // Allocate Camera Images on device for processing
    roo::Image<float, roo::TargetDevice, roo::Manage> img(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage> depth(w,h);
    roo::Image<float4, roo::TargetDevice, roo::Manage> norm(w,h);
    roo::BoundedVolume<roo::SDF_t, roo::TargetDevice, roo::Manage> vol(volres,volres,volres,make_float3(-1,-1,-1), make_float3(1,1,1));
    ActivateDrawImage<float> adg(img, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawImage<float4> adn(norm, GL_RGBA32F, true, true);

    SceneGraph::GLSceneGraph graph;
    SceneGraph::GLAxis glaxis;
    SceneGraph::GLAxisAlignedBox glbox;
    graph.AddChild(&glaxis);
    graph.AddChild(&glbox);

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h, K.fu,K.fv,K.u0,K.v0, 1,1E3),
        ModelViewLookAtRDF(0,0,-4,0,0,0,0,-1,0)
    );

    Handler3DGpuDepth handler(depth,s_cam, AxisNone);
    SceneGraph::HandlerSceneGraph handler3d(graph, s_cam, AxisNone);
    SetupContainer(container, 3, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adg)).SetHandler(&handler);
    container[1].SetDrawFunction(boost::ref(adn)).SetHandler(&handler);
    container[2].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, s_cam))
                .SetHandler( &handler3d  );

    roo::SdfSphere(vol, make_float3(0,0,0), 0.9 );

    Var<bool> subpix("ui.subpix", true, true);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        Sophus::SE3d T_cw(s_cam.GetModelViewMatrix());

        roo::RaycastSdf(depth, norm, img, vol, T_cw.inverse().matrix3x4(), K, near, far, 0, subpix );

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

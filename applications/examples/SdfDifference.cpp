#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/glvbo.h>

#include <SceneGraph/SceneGraph.h>

#include <sophus/se3.hpp>

#include <kangaroo/kangaroo.h>
#include <kangaroo/BoundedVolume.h>

#include <kangaroo/extra/ImageSelect.h>
#include <kangaroo/extra/BaseDisplayCuda.h>
#include <kangaroo/extra/DisplayUtils.h>
#include <kangaroo/extra/Handler3dGpuDepth.h>
#include <kangaroo/extra/SavePPM.h>

using namespace std;
using namespace pangolin;

int main( int argc, char* argv[] )
{
    const int w = 1024;
    const int h = 768;

    // Initialise window
    View& container = SetupPangoGLWithCuda(w, h);
    ApplyPreferredGlSettings();

    // Open video device

    const roo::ImageIntrinsics K( 570.342, 570.342, w/2.0 - 0.5, h/2.0 - 0.5 );

    roo::Image<float,  roo::TargetDevice, roo::Manage> ray_i(w,h);
    roo::Image<float,  roo::TargetDevice, roo::Manage> ray_d(w,h);
    roo::Image<float,  roo::TargetDevice, roo::Manage> ray_dist(w,h);
    roo::Image<float4, roo::TargetDevice, roo::Manage> ray_n(w,h);
    roo::Image<float4, roo::TargetDevice, roo::Manage> vis(w,h);
    roo::BoundedVolume<roo::SDF_t, roo::TargetDevice, roo::Manage> vol;
    roo::BoundedVolume<roo::SDF_t, roo::TargetDevice, roo::Manage> vol2;
    
    LoadPXM("save.vol", vol);
    LoadPXM("save2.vol", vol2);

    SceneGraph::GLSceneGraph glgraph;
    SceneGraph::GLAxisAlignedBox glboxvol;

    glboxvol.SetBounds(roo::ToEigen(vol.bbox.Min()), roo::ToEigen(vol.bbox.Max()) );
    glgraph.AddChild(&glboxvol);

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,K.fu,K.fv,K.u0,K.v0,0.1,1000),
        ModelViewLookAtRDF(0,0,-2,0,0,0,0,-1,0)
    );

    Var<float> trunc_dist_factor("ui.trunc_vol_factor",2, 1, 4);
    Var<bool> switch_sdf("ui.switch_sdf",false,true);
    Var<bool> diff_sdf("ui.diff_sdf",true,true);

    ActivateDrawImage<float4> adrayimg(vis, GL_RGBA32F, true, true);

    Handler3DGpuDepth rayhandler(ray_d, s_cam, AxisNone);
    SetupContainer(container, 2, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adrayimg))
                .SetHandler(&rayhandler);
    container[1].SetDrawFunction(SceneGraph::ActivateDrawFunctor(glgraph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );

    while(!pangolin::ShouldQuit())
    {   
        const float trunc_dist = trunc_dist_factor*length(vol.VoxelSizeUnits());

        Sophus::SE3d T_vw(s_cam.GetModelViewMatrix());
        const roo::BoundingBox roi(T_vw.inverse().matrix3x4(), w, h, K, 0, 50);
        roo::BoundedVolume<roo::SDF_t> work_vol = (switch_sdf ? vol2 : vol).SubBoundingVolume( roi );
        roo::BoundedVolume<roo::SDF_t> work_vol2 = (switch_sdf ? vol : vol2).SubBoundingVolume( roi );
        if(work_vol.IsValid()) {
            roo::RaycastSdf(ray_d, ray_n, ray_i, work_vol, T_vw.inverse().matrix3x4(), K, 0.1, 50, trunc_dist, true );
            roo::SdfDistance(ray_dist, ray_d, work_vol2, T_vw.inverse().matrix3x4(), K, trunc_dist);
            if(!diff_sdf) roo::Fill<float>(ray_dist, 0.0f);
            roo::Remap(vis, ray_i, ray_dist, -trunc_dist, trunc_dist);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);
        pangolin::FinishGlutFrame();
    }
}

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/glvbo.h>

#include <SceneGraph/SceneGraph.h>

#include <sophus/se3.hpp>

#include <kangaroo/kangaroo.h>
#include <kangaroo/BoundedVolume.h>

#include <kangaroo/common/ImageSelect.h>
#include <kangaroo/common/BaseDisplayCuda.h>
#include <kangaroo/common/DisplayUtils.h>
#include <kangaroo/common/Handler3dGpuDepth.h>
#include <kangaroo/common/SavePPM.h>

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

    const Gpu::ImageIntrinsics K( 570.342, 570.342, w/2.0 - 0.5, h/2.0 - 0.5 );

    Gpu::Image<float,  Gpu::TargetDevice, Gpu::Manage> ray_i(w,h);
    Gpu::Image<float,  Gpu::TargetDevice, Gpu::Manage> ray_d(w,h);
    Gpu::Image<float,  Gpu::TargetDevice, Gpu::Manage> ray_dist(w,h);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> ray_n(w,h);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> vis(w,h);
    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol;
    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol2;
    
    LoadPXM("save.vol", vol);
    LoadPXM("save2.vol", vol2);

    SceneGraph::GLSceneGraph glgraph;
    SceneGraph::GLAxisAlignedBox glboxvol;

    glboxvol.SetBounds(Gpu::ToEigen(vol.bbox.Min()), Gpu::ToEigen(vol.bbox.Max()) );
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
        const Gpu::BoundingBox roi(T_vw.inverse().matrix3x4(), w, h, K, 0, 50);
        Gpu::BoundedVolume<Gpu::SDF_t> work_vol = (switch_sdf ? vol2 : vol).SubBoundingVolume( roi );
        Gpu::BoundedVolume<Gpu::SDF_t> work_vol2 = (switch_sdf ? vol : vol2).SubBoundingVolume( roi );
        if(work_vol.IsValid()) {
            Gpu::RaycastSdf(ray_d, ray_n, ray_i, work_vol, T_vw.inverse().matrix3x4(), K, 0.1, 50, trunc_dist, true );
            Gpu::SdfDistance(ray_dist, ray_d, work_vol2, T_vw.inverse().matrix3x4(), K, trunc_dist);
            if(!diff_sdf) Gpu::Fill<float>(ray_dist, 0.0f);
            Gpu::Remap(vis, ray_i, ray_dist, -trunc_dist, trunc_dist);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);
        pangolin::FinishGlutFrame();
    }
}

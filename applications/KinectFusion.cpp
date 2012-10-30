#include <Eigen/Eigen>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/glvbo.h>

#include <SceneGraph/SceneGraph.h>

#include "common/ViconTracker.h"

#include <fiducials/drawing.h>

#include "common/RpgCameraOpen.h"
#include "common/ImageSelect.h"
#include "common/BaseDisplayCuda.h"
#include "common/DisplayUtils.h"
#include "common/HeightmapFusion.h"
#include "common/ViconTracker.h"
#include "common/PoseGraph.h"
#include "common/GLPoseGraph.h"
#include "common/Handler3dGpuDepth.h"
#include "common/SavePPM.h"

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>
#include <kangaroo/BoundedVolume.h>

using namespace std;
using namespace pangolin;

//Gpu::Mat<ImageKeyframe<Timg>,N> keyframes;

int main( int argc, char* argv[] )
{
    // Initialise window
    View& container = SetupPangoGLWithCuda(1024, 768);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

    // Open video device
    CameraDevice camera = OpenRpgCamera(argc,argv);
    std::vector<rpg::ImageWrapper> img;
    camera.Capture(img);
    const int w = img[0].width();
    const int h = img[0].height();
    const int MaxLevels = 4;

    const double baseline_m = camera.GetProperty<double>("Depth0Baseline", 0) / 100;
    const double fu = camera.GetProperty<double>("Depth0FocalLength", 570.342);
    const double fv = fu;
    const double u0 = w/2.0 - 0.5;
    const double v0 = h/2.0 - 0.5;
    const double knear = 0.4;
    const double kfar = 4;
//    const int volres = 384; //256;
    const int volres = 256;
    const float volrad = 1;

    // Camera (rgb) to depth
    Eigen::Vector3d c_d(baseline_m,0,0);
    Sophus::SE3 T_cd = Sophus::SE3(Sophus::SO3(),c_d).inverse();

    Gpu::Image<unsigned short, Gpu::TargetDevice, Gpu::Manage> dKinect(w,h);
    Gpu::Pyramid<float, MaxLevels, Gpu::TargetDevice, Gpu::Manage> kin_d(w,h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> kin_v(w,h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> kin_n(w,h);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage>  dDebug(w,h);
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> dScratch(w*sizeof(Gpu::LeastSquaresSystem<float,12>),h);

    Gpu::Pyramid<float, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_i(w,h);
    Gpu::Pyramid<float, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_d(w,h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_n(w,h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_v(w,h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_c(w,h);
//    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(volres,volres,volres,make_float3(-volrad,-volrad,-volrad), make_float3(volrad,volrad,volrad));
    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(volres,volres,volres,make_float3(-0.25,-0.5,0.75), make_float3(0.25,0.5,1.25)); // dress form.
    const float3 voxsize = vol.VoxelSizeUnits();

    Gpu::Image<uchar3, Gpu::TargetDevice, Gpu::Manage> kfrgb(w,h);

    Gpu::ImageKeyframe<uchar3> kf;
    kf.img = kfrgb;

    SceneGraph::GLSceneGraph glgraph;
    SceneGraph::GLAxis glcamera(0.1);
    SceneGraph::GLAxisAlignedBox glboxfrustum;
    SceneGraph::GLAxisAlignedBox glboxvol;

    glboxvol.SetBounds(Gpu::ToEigen(vol.bbox.Min()), Gpu::ToEigen(vol.bbox.Max()) );
    glgraph.AddChild(&glcamera);
    glgraph.AddChild(&glboxvol);
    glgraph.AddChild(&glboxfrustum);

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,fu,fv,u0,v0,0.1,1000),
        ModelViewLookAtRDF(0,0,-2,0,0,0,0,-1,0)
    );

    Var<bool> run("ui.run", true, true);
    Var<bool> viewonly("ui.view only", false, true);
    Var<int> show_level("ui.Show Level", 2, 0, MaxLevels-1);

    Var<int> biwin("ui.size",5, 1, 20);
    Var<float> bigs("ui.gs",5, 1E-3, 5);
    Var<float> bigr("ui.gr",700, 1E-3, 200);

    Var<bool> pose_refinement("ui.Pose Refinement", true, true);
    Var<bool> reset("ui.reset", false, false);
    Var<float> icp_c("ui.icp c",0.5, 1E-3, 1);
    Var<int> pose_its("ui.pose_its", 5, 0, 10);

    Var<bool> fuse("ui.fuse", true, true);
    Var<bool> fuseonce("ui.fuse once", false, false);

    Var<float> trunc_dist("ui.trunc dist", 2*length(voxsize), 2*length(voxsize),0.5);
    Var<float> max_w("ui.max w", 10, 1E-4, 10);
    Var<float> mincostheta("ui.min cos theta", 0.1, 0, 1);

    Var<bool> save_kf("ui.Save KF", false, false);
    Var<float> rgb_fl("ui.RGB focal length", 535.7,400,600);

    Eigen::Matrix3d Krgb;
    Krgb << rgb_fl, 0, w/2.0f - 0.5,
            0, rgb_fl, h/2.0f - 0.5,
            0,0,1;

    ActivateDrawPyramid<float,MaxLevels> adrayimg(ray_i, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawPyramid<float4,MaxLevels> adraycolor(ray_c, GL_RGBA32F, true, true);
//    ActivateDrawPyramid<float4,MaxLevels> adraynorm(ray_n, GL_RGBA32F, true, true);
//    ActivateDrawPyramid<float,MaxLevels> addepth( kin_d, GL_LUMINANCE32F_ARB, false, true);
//    ActivateDrawPyramid<float4,MaxLevels> adnormals( kin_n, GL_RGBA32F_ARB, false, true);
    ActivateDrawImage<float4> addebug( dDebug, GL_RGBA32F_ARB, false, true);

    Handler3DGpuDepth rayhandler(ray_d[0], s_cam, AxisNone);
    SetupContainer(container, 3, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adrayimg))
                .SetHandler(&rayhandler);
    container[1].SetDrawFunction(SceneGraph::ActivateDrawFunctor(glgraph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );
    container[2].SetDrawFunction(boost::ref(adraycolor))
                .SetHandler(&rayhandler);
//    container[1].SetDrawFunction(boost::ref(adnormals));
//    container[0].SetDrawFunction(boost::ref(addebug));
//    container[4].SetDrawFunction(boost::ref(adraynorm));

    Sophus::SE3 T_wl;

//    pangolin::RegisterKeyPressCallback(' ', [&posegraph]() {posegraph.Start();} );

    for(long frame=-1; !pangolin::ShouldQuit();)
    {
        const bool go = frame==-1 || run;

        if(rgb_fl.GuiChanged()) {
            Krgb << rgb_fl, 0, w/2.0f - 0.5,
                    0, rgb_fl, h/2.0f - 0.5,
                    0,0,1;
            save_kf = true;
        }

        if(Pushed(save_kf)) {
            Eigen::Matrix<double,3,4> KiTiw = Krgb * (T_cd * T_wl.inverse()).matrix3x4();
            kf.img.CopyFrom(Gpu::Image<uchar3, Gpu::TargetHost>((uchar3*)img[0].Image.data,w,h));
            kf.KiT_iw = KiTiw;
        }

        if(go) {
            camera.Capture(img);
            dKinect.CopyFrom(Gpu::Image<unsigned short, Gpu::TargetHost>((unsigned short*)img[1].Image.data,w,h));
            Gpu::BilateralFilter<float,unsigned short>(kin_d[0],dKinect,bigs,bigr,biwin,200);
            Gpu::ElementwiseScaleBias<float,float,float>(kin_d[0], kin_d[0], 1.0f/1000.0f);

            Gpu::BoxReduceIgnoreInvalid<float,MaxLevels,float>(kin_d);
            for(int l=0; l<MaxLevels; ++l) {
                Gpu::DepthToVbo(kin_v[l], kin_d[l], fu/(1<<l), fv/(1<<l), w/(2.0f * (1<<l)) - 0.5, h/(2.0f * (1<<l)) - 0.5 );
                Gpu::NormalsFromVbo(kin_n[l], kin_v[l]);
            }

            frame++;
        }

        if(Pushed(reset) || frame==0) {
            T_wl = Sophus::SE3();
            Gpu::SdfReset(vol, trunc_dist);

            // Fuse first kinect frame in.
            Gpu::SdfFuse(vol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), fu, fv, u0, v0, trunc_dist, max_w, mincostheta );
        }

        if(viewonly) {
            Sophus::SE3 T_vw(s_cam.GetModelViewMatrix());
            Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( Gpu::BoundingBox(T_vw.inverse().matrix3x4(), w, h, fu, fv, u0, v0, knear,20) );
            Gpu::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_vw.inverse().matrix3x4(), fu, fv, u0, v0, knear,20, true );
            Gpu::TextureDepth<float4,uchar3>(ray_c[0], kf, ray_d[0], T_vw.inverse().matrix3x4(), fu,fv,u0,v0);
        }else{
            if(pose_refinement && frame > 0) {
                Sophus::SE3 T_lp;
                Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( Gpu::BoundingBox(T_wl.matrix3x4(), w, h, fu, fv, u0, v0, knear,kfar) );

    //            for(int l=MaxLevels-1; l >=0; --l)
                {
                    const int l = show_level;


                    Gpu::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, T_wl.matrix3x4(), fu/(1<<l), fv/(1<<l), w/(2 * 1<<l) - 0.5, h/(2 * 1<<l) - 0.5, knear,kfar, true );
                    Gpu::DepthToVbo(ray_v[l], ray_d[l], fu/(1<<l), fv/(1<<l), w/(2.0f * (1<<l)) - 0.5, h/(2.0f * (1<<l)) - 0.5 );

                    const int lits = pose_its;
                    Eigen::Matrix3d Kdepth;
                    Kdepth << fu/(1<<l), 0, w/(2.0f * (1<<l)) - 0.5,   0, fv/(1<<l), h/(2.0f * (1<<l)) - 0.5,  0,0,1;

                    for(int i=0; i<lits; ++i ) {
                        const Eigen::Matrix<double, 3,4> mKT_lp = Kdepth * T_lp.matrix3x4();
                        const Eigen::Matrix<double, 3,4> mT_pl = T_lp.inverse().matrix3x4();
                        Gpu::LeastSquaresSystem<float,6> lss = Gpu::PoseRefinementProjectiveIcpPointPlane(
                            kin_v[l], ray_v[l], ray_n[l], mKT_lp, mT_pl, icp_c, dScratch, dDebug.SubImage(0,0,w>>l,h>>l)
                        );
                        Eigen::FullPivLU<Eigen::Matrix<double,6,6> > lu_JTJ( (Eigen::Matrix<double,6,6>)lss.JTJ );
                        Eigen::Matrix<double,6,1> x = -1.0 * lu_JTJ.solve( (Eigen::Matrix<double,6,1>)lss.JTy );
                        T_lp = T_lp * Sophus::SE3::exp(x);
                    }
                }

                T_wl = T_wl * T_lp.inverse();
            }

            if(pose_refinement && (Pushed(fuseonce) || fuse) ) {
                Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( Gpu::BoundingBox(T_wl.matrix3x4(), w, h, fu, fv, u0, v0, knear,kfar) );
                Gpu::SdfFuse(work_vol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), fu, fv, u0, v0, trunc_dist, max_w, mincostheta );
            }
        }

        glcamera.SetPose(T_wl.matrix());

        Gpu::BoundingBox bbox_work(T_wl.matrix3x4(), w, h, fu, fv, u0, v0, knear,kfar);
        bbox_work.Intersect(vol.bbox);
        glboxfrustum.SetBounds(Gpu::ToEigen(bbox_work.Min()), Gpu::ToEigen(bbox_work.Max()) );

//        {
//            CudaScopedMappedPtr var(cbo);
//            Gpu::Image<uchar4> dCbo((uchar4*)*var,w,h);
//            Gpu::ConvertImage<uchar4,float4>(dCbo,kin_n[0]);
//        }

//        {
//            CudaScopedMappedPtr var(vbo);
//            Gpu::Image<float4> dVbo((float4*)*var,w,h);
//            dVbo.CopyFrom(kin_v[0]);
//        }

        /////////////////////////////////////////////////////////////
        // Draw
        addebug.SetImage(dDebug.SubImage(0,0,w>>show_level,h>>show_level));
//        addepth.SetImageScale(scale);
//        addepth.SetLevel(show_level);
//        adnormals.SetLevel(show_level);
        adrayimg.SetLevel(viewonly? 0 : show_level);
//        adraynorm.SetLevel(show_level);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);
        pangolin::FinishGlutFrame();
    }
}

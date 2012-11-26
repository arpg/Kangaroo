#include <Eigen/Eigen>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/glvbo.h>

#include <CVars/CVar.h>

#include <SceneGraph/SceneGraph.h>

#include "common/ViconTracker.h"

#include <fiducials/drawing.h>

#include "common/RpgCameraOpen.h"
#include "common/ImageSelect.h"
#include "common/BaseDisplayCuda.h"
#include "common/DisplayUtils.h"
#include "common/ViconTracker.h"
#include "common/PoseGraph.h"
#include "common/GLPoseGraph.h"
#include "common/Handler3dGpuDepth.h"
#include "common/SavePPM.h"
#include "common/SaveGIL.h"
#include "common/SaveMeshlab.h"
#include "common/CVarHelpers.h"

#include "MarchingCubes.h"

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>
#include <kangaroo/BoundedVolume.h>

using namespace std;
using namespace pangolin;

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
    const float volrad = 0.4;

    Gpu::BoundingBox reset_bb(make_float3(-volrad,-volrad,0.5), make_float3(volrad,volrad,0.5+2*volrad));
//    Gpu::BoundingBox reset_bb(make_float3(-volrad,-volrad,-volrad), make_float3(volrad,volrad,volrad));
//    Gpu::BoundingBox reset_bb(make_float3(-0.25,-0.5,0.75), make_float3(0.25,0.5,1.25));

    CVarUtils::AttachCVar<Gpu::BoundingBox>("BoundingBox", &reset_bb);

    const Eigen::Vector4d its(1,0,2,3);

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
    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(volres,volres,volres,reset_bb); // in front

    boost::ptr_vector<KinectKeyframe> keyframes;
    Gpu::Mat<Gpu::ImageKeyframe<uchar3>,10> kfs;

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
    Var<bool> fuse("ui.fuse", true, true);
    Var<bool> reset("ui.reset", true, false);

    Var<int> show_level("ui.Show Level", 0, 0, MaxLevels-1);

    Var<int> biwin("ui.size",5, 1, 20);
    Var<float> bigs("ui.gs",5, 1E-3, 5);
    Var<float> bigr("ui.gr",100, 1E-3, 200);

    Var<bool> pose_refinement("ui.Pose Refinement", true, true);
    Var<float> icp_c("ui.icp c",0.1, 1E-3, 1);

    Var<float> max_w("ui.max w", 100, 1E-4, 10);
    Var<float> mincostheta("ui.min cos theta", 0.1, 0, 1);

    Var<bool> save_kf("ui.Save KF", false, false);
    Var<float> rgb_fl("ui.RGB focal length", 535.7,400,600);
    Var<float> max_rmse("ui.Max RMSE",0.10,0,0.5);
    Var<float> rmse("ui.RMSE",0);

    ActivateDrawPyramid<float,MaxLevels> adrayimg(ray_i, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawPyramid<float4,MaxLevels> adraycolor(ray_c, GL_RGBA32F, true, true);
    ActivateDrawPyramid<float4,MaxLevels> adraynorm(ray_n, GL_RGBA32F, true, true);
//    ActivateDrawPyramid<float,MaxLevels> addepth( kin_d, GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawPyramid<float4,MaxLevels> adnormals( kin_n, GL_RGBA32F_ARB, false, true);
    ActivateDrawImage<float4> addebug( dDebug, GL_RGBA32F_ARB, false, true);

    Handler3DGpuDepth rayhandler(ray_d[0], s_cam, AxisNone);
    SetupContainer(container, 3, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adrayimg))
                .SetHandler(&rayhandler);
    container[1].SetDrawFunction(SceneGraph::ActivateDrawFunctor(glgraph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );
    container[2].SetDrawFunction(boost::ref(adraycolor))
                .SetHandler(&rayhandler);
//    container[3].SetDrawFunction(boost::ref(addebug));
//    container[4].SetDrawFunction(boost::ref(adnormals));
//    container[5].SetDrawFunction(boost::ref(adraynorm));

    Sophus::SE3 T_wl;

    pangolin::RegisterKeyPressCallback('l', [&vol,&viewonly]() {LoadPXM("save.vol", vol); viewonly = true;} );
    pangolin::RegisterKeyPressCallback('s', [&vol,&keyframes,&rgb_fl,w,h]() {SavePXM("save.vol", vol); SaveMeshlab(vol,keyframes,rgb_fl,rgb_fl,w/2,h/2); } );

    for(long frame=-1; !pangolin::ShouldQuit();)
    {
        const bool go = frame==-1 || run;

        if(Pushed(save_kf)) {
            KinectKeyframe* kf = new KinectKeyframe(w,h,T_cd * T_wl.inverse());
            kf->img.CopyFrom(Gpu::Image<uchar3, Gpu::TargetHost>((uchar3*)img[0].Image.data,w,h));
            keyframes.push_back(kf);
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

        if(Pushed(reset)) {
            T_wl = Sophus::SE3();

            vol.bbox = reset_bb;
            Gpu::SdfReset(vol);
            keyframes.clear();

            // Fuse first kinect frame in.
            const float trunc_dist = 2*length(vol.VoxelSizeUnits());
            Gpu::SdfFuse(vol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), fu, fv, u0, v0, trunc_dist, max_w, mincostheta );
        }

        if(viewonly) {
            const float trunc_dist = 2*length(vol.VoxelSizeUnits());

            Sophus::SE3 T_vw(s_cam.GetModelViewMatrix());
            Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( Gpu::BoundingBox(T_vw.inverse().matrix3x4(), w, h, fu, fv, u0, v0, 0, 20) );
            if(work_vol.IsValid()) {
                Gpu::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_vw.inverse().matrix3x4(), fu, fv, u0, v0, 0.1, 20, trunc_dist, true );

                // populate kfs
                for( int k=0; k< kfs.Rows(); k++)
                {
                    if(k < keyframes.size()) {
                        kfs[k].img = keyframes[k].img;
                        kfs[k].T_iw = keyframes[k].T_iw.matrix3x4();
                        kfs[k].K = Gpu::ImageIntrinsics(rgb_fl, kfs[k].img);
                    }else{
                        kfs[k].img.ptr = 0;
                    }
                }

                Gpu::TextureDepth<float4,uchar3,10>(ray_c[0], kfs, ray_d[0], ray_n[0], ray_i[0], T_vw.inverse().matrix3x4(), fu,fv,u0,v0);
            }
        }else{
            bool tracking_good = true;

            Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( Gpu::BoundingBox(T_wl.matrix3x4(), w, h, fu, fv, u0, v0, knear,kfar) );
            if(work_vol.IsValid()) {
//                Gpu::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_wl.matrix3x4(), fu, fv, u0, v0, knear, kfar, true );
//                Gpu::BoxReduceIgnoreInvalid<float,MaxLevels,float>(ray_d);
                for(int l=0; l<MaxLevels; ++l) {
                    if(its[l] > 0) {
                        Gpu::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, T_wl.matrix3x4(), fu/(1<<l), fv/(1<<l), w/(2 * 1<<l) - 0.5, h/(2 * 1<<l) - 0.5, knear,kfar, true );
                        Gpu::DepthToVbo(ray_v[l], ray_d[l], fu/(1<<l), fv/(1<<l), w/(2.0f * (1<<l)) - 0.5, h/(2.0f * (1<<l)) - 0.5 );
    //                    Gpu::DepthToVbo(ray_v[l], ray_d[l], fu/(1<<l), fv/(1<<l), w/(2.0f * (1<<l)) - 0.5, h/(2.0f * (1<<l)) - 0.5 );
    //                    Gpu::NormalsFromVbo(ray_n[l], ray_v[l]);
                    }
                }

                if(pose_refinement && frame > 0) {
                    Sophus::SE3 T_lp;

//                    const int l = show_level;
//                    Gpu::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, T_wl.matrix3x4(), fu/(1<<l), fv/(1<<l), w/(2 * 1<<l) - 0.5, h/(2 * 1<<l) - 0.5, knear,kfar, true );
//                    Gpu::DepthToVbo(ray_v[l], ray_d[l], fu/(1<<l), fv/(1<<l), w/(2.0f * (1<<l)) - 0.5, h/(2.0f * (1<<l)) - 0.5 );

                    for(int l=MaxLevels-1; l >=0; --l)
                    {
                        Eigen::Matrix3d Kdepth;
                        Kdepth << fu/(1<<l), 0, w/(2.0f * (1<<l)) - 0.5,   0, fv/(1<<l), h/(2.0f * (1<<l)) - 0.5,  0,0,1;

                        for(int i=0; i<its[l]; ++i ) {
                            const Eigen::Matrix<double, 3,4> mKT_lp = Kdepth * T_lp.matrix3x4();
                            const Eigen::Matrix<double, 3,4> mT_pl = T_lp.inverse().matrix3x4();
                            Gpu::LeastSquaresSystem<float,6> lss = Gpu::PoseRefinementProjectiveIcpPointPlane(
                                kin_v[l], ray_v[l], ray_n[l], mKT_lp, mT_pl, icp_c, dScratch, dDebug.SubImage(0,0,w>>l,h>>l)
                            );

                            Eigen::Matrix<double,6,6> sysJTJ = lss.JTJ;
                            Eigen::Matrix<double,6,1> sysJTy = lss.JTy;

                            // Add a week prior on our pose
                            const double motionSigma = 0.2;
                            const double depthSigma = 0.1;
                            sysJTJ += (depthSigma / motionSigma) * Eigen::Matrix<double,6,6>::Identity();

                            rmse = sqrt(lss.sqErr / lss.obs);
                            tracking_good = rmse < max_rmse;

                            if(l == MaxLevels-1) {
                                // Solve for rotation only
                                Eigen::FullPivLU<Eigen::Matrix<double,3,3> > lu_JTJ( sysJTJ.block<3,3>(3,3) );
                                Eigen::Matrix<double,3,1> x = -1.0 * lu_JTJ.solve( sysJTy.segment<3>(3) );
                                T_lp = T_lp * Sophus::SE3(Sophus::SO3::exp(x), Eigen::Vector3d(0,0,0) );
                            }else{
                                Eigen::FullPivLU<Eigen::Matrix<double,6,6> > lu_JTJ( sysJTJ );
                                Eigen::Matrix<double,6,1> x = -1.0 * lu_JTJ.solve( sysJTy );
                                T_lp = T_lp * Sophus::SE3::exp(x);
                            }

                        }
                    }

                    T_wl = T_wl * T_lp.inverse();
                }
            }

            if(pose_refinement && fuse) {
                if(tracking_good) {
                    Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( Gpu::BoundingBox(T_wl.matrix3x4(), w, h, fu, fv, u0, v0, knear,kfar) );
                    if(work_vol.IsValid()) {
                        const float trunc_dist = 2*length(vol.VoxelSizeUnits());
                        Gpu::SdfFuse(work_vol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), fu, fv, u0, v0, trunc_dist, max_w, mincostheta );
                    }
                }else{
                    cerr << "Tracking bad" << endl;
                }
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
        adnormals.SetLevel(show_level);
        adrayimg.SetLevel(viewonly? 0 : show_level);
//        adraynorm.SetLevel(show_level);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);
        pangolin::FinishGlutFrame();
    }
}


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

struct KinectRgbdKeyframe
{
    KinectRgbdKeyframe(int w, int h, Sophus::SE3 Twi)
        : img_d(w,h),img_rgb(w,h), T_wi(Twi)
    {
    }

    Sophus::SE3 T_wi;
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> img_d;
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> img_rgb;
};

int main( int argc, char* argv[] )
{
    // Initialise window
    View& container = SetupPangoGLWithCuda(1024, 768);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

    // Open video device
    CameraDevice camera = OpenRpgCamera(argc,argv);
    std::vector<rpg::ImageWrapper> img;
    camera.Capture(img);
    const int image_w = img[0].width();
    const int image_h = img[0].height();
    const int MaxLevels = 4;

    const double baseline_m = camera.GetProperty<double>("Depth0Baseline", 0) / 100;
    const Gpu::ImageIntrinsics K(
        camera.GetProperty<double>("Depth0FocalLength", 570.342),
        camera.GetProperty<double>("Depth0FocalLength", 570.342),
        image_w/2.0 - 0.5, image_h/2.0 - 0.5
    );
    const double knear = 0.4;
    const double kfar = 4;
//    const int volres = 384; //256;
    const int volres = 256;
    const float volrad = 1;

    Gpu::BoundingBox reset_bb(make_float3(-volrad,-volrad,0.5), make_float3(volrad,volrad,0.5+2*volrad));
//    Gpu::BoundingBox reset_bb(make_float3(-volrad,-volrad,-volrad), make_float3(volrad,volrad,volrad));

    CVarUtils::AttachCVar<Gpu::BoundingBox>("BoundingBox", &reset_bb);

    const Eigen::Vector4d its(1,0,2,3);

    // Camera (rgb) to depth
    Eigen::Vector3d c_d(baseline_m,0,0);
    Sophus::SE3 T_cd = Sophus::SE3(Sophus::SO3(),c_d).inverse();

    Gpu::Image<unsigned short, Gpu::TargetDevice, Gpu::Manage> dKinect(image_w,image_h);
    Gpu::Image<uchar3, Gpu::TargetDevice, Gpu::Manage> drgb(image_w,image_h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> dKinectMeters(image_w,image_h);
    Gpu::Pyramid<float, MaxLevels, Gpu::TargetDevice, Gpu::Manage> kin_d(image_w,image_h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> kin_v(image_w,image_h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> kin_n(image_w,image_h);
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage>  dImage(image_w,image_h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage>  dDepth(image_w,image_h);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage>  dDebug(image_w,image_h);
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> dScratch(image_w*sizeof(Gpu::LeastSquaresSystem<float,12>),image_h);

    Gpu::Pyramid<float, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_i(image_w,image_h);
    Gpu::Pyramid<float, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_d(image_w,image_h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_n(image_w,image_h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_v(image_w,image_h);
    Gpu::Pyramid<float4, MaxLevels, Gpu::TargetDevice, Gpu::Manage> ray_c(image_w,image_h);
    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(volres,volres,volres,reset_bb);
    Gpu::BoundedVolume<float, Gpu::TargetDevice, Gpu::Manage> colorVol(volres,volres,volres,reset_bb);

    //workspace for LSS
    Gpu::Image< unsigned char, Gpu::TargetDevice, Gpu::Manage > lss_workspace( image_w * sizeof(Gpu::LeastSquaresSystem<float,6>), image_h );
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> lss_debug(image_w,image_h);

    boost::ptr_vector<KinectRgbdKeyframe> keyframes;
    Eigen::Matrix4d T_di = Eigen::Matrix4d::Identity();
//     set at solution
    //Eigen::Matrix4d T_di = T_cd.inverse().matrix();
    // Kinect T
    // 0.0252783  0.00240222 -0.00611408    -0.26625   -0.265614   -0.263682
    // 0.0232425 0.00899586 0.00497145  -0.435737  -0.441952  -0.452373

    SceneGraph::GLSceneGraph glgraph;
    SceneGraph::GLAxis glcamera(0.1);
    SceneGraph::GLAxisAlignedBox glboxfrustum;
    SceneGraph::GLAxisAlignedBox glboxvol;

    glboxvol.SetBounds(Gpu::ToEigen(vol.bbox.Min()), Gpu::ToEigen(vol.bbox.Max()) );
    glgraph.AddChild(&glcamera);
    glgraph.AddChild(&glboxvol);
    glgraph.AddChild(&glboxfrustum);

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(image_w,image_h,K.fu,K.fv,K.u0,K.v0,0.1,1000),
        ModelViewLookAtRDF(0,0,-2,0,0,0,0,-1,0)
    );

    Var<bool> run("ui.run", true, true);

    Var<bool> showcolor("ui.show color", false, true);
    Var<bool> viewonly("ui.view only", false, true);
    Var<bool> fuse("ui.fuse", true, true);
    Var<bool> reset("ui.reset", true, false);

    Var<int> show_level("ui.Show Level", 0, 0, MaxLevels-1);

    // TODO: This needs to be a function of the inverse depth
    Var<int> biwin("ui.size",3, 1, 20);
    Var<float> bigs("ui.gs",1.5, 1E-3, 5);
    Var<float> bigr("ui.gr",0.1, 1E-6, 0.2);

    Var<bool> pose_refinement("ui.Pose Refinement", true, true);
    Var<float> icp_c("ui.icp c",0.1, 1E-3, 1);
    Var<float> trunc_dist_factor("ui.trunc vol factor",2, 1, 4);

    Var<float> max_w("ui.max w", 1000, 1E-2, 1E3);
    Var<float> mincostheta("ui.min cos theta", 0.1, 0, 1);

    Var<bool> save_kf("ui.Save KF", false, false);
    Var<float> rgb_fl("ui.RGB focal length", 535.7,400,600);
    Var<float> max_rmse("ui.Max RMSE",0.10,0,0.5);
    Var<float> rmse("ui.RMSE",0);
    Var<bool> bCalibrate("ui.Calibrate",0);
    Var<bool> bStep("ui.Step",0);

    ActivateDrawPyramid<float,MaxLevels> adrayimg(ray_i, GL_LUMINANCE32F_ARB, true, true);
//    ActivateDrawPyramid<float,MaxLevels> addepth( kin_d, GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawPyramid<float4,MaxLevels> adnormals( kin_n, GL_RGBA32F_ARB, false, true);
    ActivateDrawImage<unsigned char> adImage( dImage, GL_LUMINANCE8, false, true);
    ActivateDrawImage<float> adDepth( dDepth, GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float4> addebug( lss_debug, GL_RGBA32F_ARB, false, true);

    Handler3DGpuDepth rayhandler(ray_d[0], s_cam, AxisNone);
    SetupContainer(container, 5, (float)image_w/image_h);
    container[0].SetDrawFunction(boost::ref(adrayimg))
                .SetHandler(&rayhandler);
    container[1].SetDrawFunction(SceneGraph::ActivateDrawFunctor(glgraph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );
    container[2].SetDrawFunction(boost::ref(addebug));
    container[3].SetDrawFunction(boost::ref(adImage));
    container[4].SetDrawFunction(boost::ref(adDepth));

    Sophus::SE3 T_wl;

    pangolin::RegisterKeyPressCallback(' ', [&reset,&viewonly]() { reset = true; viewonly=false;} );
    pangolin::RegisterKeyPressCallback('l', [&vol,&viewonly]() {LoadPXM("save.vol", vol); viewonly = true;} );
//    pangolin::RegisterKeyPressCallback('s', [&vol,&colorVol,&keyframes,&rgb_fl,w,h]() {SavePXM("save.vol", vol); SaveMeshlab(vol,keyframes,rgb_fl,rgb_fl,w/2,h/2); } );
    pangolin::RegisterKeyPressCallback('s', [&vol,&colorVol]() {Gpu::SaveMesh("mesh",vol,colorVol); } );
    //this will capture a keyframe
    pangolin::RegisterKeyPressCallback('k', [&save_kf]() { save_kf = true; } );
    pangolin::RegisterKeyPressCallback('c', [&bCalibrate]() { bCalibrate = true; } );
    pangolin::RegisterKeyPressCallback('s', [&bStep]() { bStep = true; } );


    for(long frame=-1; !pangolin::ShouldQuit();)
    {
        const bool go = !viewonly && (frame==-1 || run);



        if(go) {
            camera.Capture(img);
            dKinect.CopyFrom(Gpu::Image<unsigned short, Gpu::TargetHost>((unsigned short*)img[1].Image.data,image_w,image_h));
            drgb.CopyFrom(Gpu::Image<uchar3, Gpu::TargetHost>((uchar3*)img[0].Image.data,image_w,image_h));
            Gpu::ElementwiseScaleBias<float,unsigned short,float>(dKinectMeters, dKinect, 1.0f/1000.0f);
            Gpu::BilateralFilter<float,float>(kin_d[0],dKinectMeters,bigs,bigr,biwin,0.2);

            Gpu::BoxReduceIgnoreInvalid<float,MaxLevels,float>(kin_d);
            for(int l=0; l<MaxLevels; ++l) {
                Gpu::DepthToVbo(kin_v[l], kin_d[l], K[l] );
                Gpu::NormalsFromVbo(kin_n[l], kin_v[l]);
            }

            frame++;
        }

        if(Pushed(reset)) {
            T_wl = Sophus::SE3();

            vol.bbox = reset_bb;
            Gpu::SdfReset(vol);
            keyframes.clear();

            colorVol.bbox = reset_bb;
            Gpu::SdfReset(colorVol);

            // Fuse first kinect frame in.
            const float trunc_dist = trunc_dist_factor*length(vol.VoxelSizeUnits());
//            Gpu::SdfFuse(vol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), K, trunc_dist, max_w, mincostheta );
            Gpu::SdfFuse(vol, colorVol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), K, drgb, (T_cd * T_wl.inverse()).matrix3x4(), Gpu::ImageIntrinsics(rgb_fl, drgb), trunc_dist, max_w, mincostheta );
        }

        if(viewonly) {
            const float trunc_dist = trunc_dist_factor*length(vol.VoxelSizeUnits());

            Sophus::SE3 T_vw(s_cam.GetModelViewMatrix());
            const Gpu::BoundingBox roi(T_vw.inverse().matrix3x4(), image_w, image_h, K, 0, 50);
            Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( roi );
            Gpu::BoundedVolume<float> work_colorVol = colorVol.SubBoundingVolume( roi );
            if(work_vol.IsValid()) {
                if(showcolor) {
                    Gpu::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, work_colorVol, T_vw.inverse().matrix3x4(), K, 0.1, 50, trunc_dist, true );
                }else{
                    Gpu::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_vw.inverse().matrix3x4(), K, 0.1, 50, trunc_dist, true );
                }


//                if(keyframes.size() > 0) {
//                    // populate kfs
//                    for( int k=0; k< kfs.Rows(); k++)
//                    {
//                        if(k < keyframes.size()) {
//                            kfs[k].img = keyframes[k].img;
//                            kfs[k].T_iw = keyframes[k].T_iw.matrix3x4();
//                            kfs[k].K = Gpu::ImageIntrinsics(rgb_fl, kfs[k].img);
//                        }else{
//                            kfs[k].img.ptr = 0;
//                        }
//                    }
//                    Gpu::TextureDepth<float4,uchar3,10>(ray_c[0], kfs, ray_d[0], ray_n[0], ray_i[0], T_vw.inverse().matrix3x4(), K);
//                }
            }
        } else {
            bool tracking_good = true;

            const Gpu::BoundingBox roi(Gpu::BoundingBox(T_wl.matrix3x4(), image_w, image_h, K, knear,kfar));
            Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( roi );
            Gpu::BoundedVolume<float> work_colorVol = colorVol.SubBoundingVolume( roi );
            if(work_vol.IsValid()) {
//                Gpu::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_wl.matrix3x4(), fu, fv, u0, v0, knear, kfar, true );
//                Gpu::BoxReduceIgnoreInvalid<float,MaxLevels,float>(ray_d);
                for(int l=0; l<MaxLevels; ++l) {
                    if(its[l] > 0) {
                        const Gpu::ImageIntrinsics Kl = K[l];
                        if(showcolor) {
                            Gpu::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, colorVol, T_wl.matrix3x4(), Kl, knear,kfar, true );
                        }else{
                            Gpu::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, T_wl.matrix3x4(), Kl, knear,kfar, true );
                        }
                        Gpu::DepthToVbo(ray_v[l], ray_d[l], Kl );
    //                    Gpu::DepthToVbo(ray_v[l], ray_d[l], Kl.fu, Kl.fv, Kl.u0, Kl.v0 );
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
                        const Eigen::Matrix3d Kdepth = K[l].Matrix();

                        for(int i=0; i<its[l]; ++i ) {
                            const Eigen::Matrix<double, 3,4> mKT_lp = Kdepth * T_lp.matrix3x4();
                            const Eigen::Matrix<double, 3,4> mT_pl = T_lp.inverse().matrix3x4();
                            Gpu::LeastSquaresSystem<float,6> lss = Gpu::PoseRefinementProjectiveIcpPointPlane(
                                kin_v[l], ray_v[l], ray_n[l], mKT_lp, mT_pl, icp_c, dScratch, dDebug.SubImage(0,0,image_w>>l,image_h>>l)
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
                            } else {
                                Eigen::FullPivLU<Eigen::Matrix<double,6,6> > lu_JTJ( sysJTJ );
                                Eigen::Matrix<double,6,1> x = -1.0 * lu_JTJ.solve( sysJTy );
                                T_lp = T_lp * Sophus::SE3::exp(x);
                            }

                        }
                    }

                    if(tracking_good) {
                        T_wl = T_wl * T_lp.inverse();
                    }
                }
            }

            if(pose_refinement && fuse) {
                if(tracking_good) {
                    const Gpu::BoundingBox roi(T_wl.matrix3x4(), image_w, image_h, K, knear,kfar);
                    Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( roi );
                    Gpu::BoundedVolume<float> work_colorVol = colorVol.SubBoundingVolume( roi );
                    if(work_vol.IsValid()) {
                        const float trunc_dist = trunc_dist_factor*length(vol.VoxelSizeUnits());
//                        Gpu::SdfFuse(work_vol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), K, trunc_dist, max_w, mincostheta );
                        Gpu::SdfFuse(work_vol, work_colorVol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), K, drgb, (T_cd * T_wl.inverse()).matrix3x4(), Gpu::ImageIntrinsics(rgb_fl, drgb), trunc_dist, max_w, mincostheta );
                    }
                }
            }

            if(pose_refinement && fuse == false){
                //if the user wants to capture a keyframe, store the rgb and depth images into the keyframe struct
                if(Pushed(save_kf)) {
                    KinectRgbdKeyframe* kf = new KinectRgbdKeyframe(image_w,image_h,T_wl);
                    drgb.MemcpyFromHost(img[0].Image.data);
                    Gpu::ConvertImage<unsigned char,uchar3>(kf->img_rgb, drgb);
                    kf->img_d.CopyFrom(ray_d[0]);
                    keyframes.push_back(kf);
                    std::cout << "Pushing keyframe number " << keyframes.size() << std::endl;
                }

                if(bCalibrate == true && bStep == true){
                    bStep = false;
                    std::cout << "Solving for calibration..." << std::endl;
                    //run the calibration
                    //the least squares system used for the optimization
                    Gpu::LeastSquaresSystem<float,6> lss;
                    lss.SetZero();
                    //for(int kk = 0 ; kk < 10 ; kk++){
//                        std::cout << "Preparing iteration " << kk << std::endl;
                        for(int ii = 0 ; ii < keyframes.size() ; ii++){
                            for(int jj = 0 ; jj < keyframes.size() ; jj++){
                                if(ii!= jj){
                                    const KinectRgbdKeyframe& keyframe_r = keyframes[ii];
                                    const KinectRgbdKeyframe& keyframe_l = keyframes[jj];
                                    const Sophus::SE3 T_wr = keyframe_r.T_wi;
                                    const Sophus::SE3 T_wl = keyframe_l.T_wi;
                                    const Sophus::SE3 T_lr = T_wl.inverse() * T_wr;
                                    Eigen::Matrix<float,3,4> eigen_fT_di = T_di.block<3,4>(0,0).cast<float>();
                                    Eigen::Matrix<float,3,3> eigen_fK = K.Matrix().cast<float>();
                                    Gpu::Mat<float,3,4> fT_di = eigen_fT_di;
                                    Gpu::Mat<float,3,3> fK = eigen_fK;


                                    // build system
                                    lss = lss + Gpu::CalibrationRgbdFromDepthESM(keyframe_l.img_rgb,keyframe_r.img_rgb,keyframe_r.img_d,
                                                                                 fK, fT_di,
                                                                                 T_lr.matrix3x4(), 10, K.fu, K.fv, K.u0, K.v0, lss_workspace,
                                                                                 lss_debug,false, 0.3, 30.0 );
                                }
                            }
                        }


                        //solve the system
                        Eigen::Matrix<double,6,6>   LHS = lss.JTJ;
                        Eigen::Vector6d             RHS = lss.JTy;
                        Eigen::Vector6d             X;

                        Eigen::FullPivLU<Eigen::Matrix<double,6,6> >    lu_JTJ(LHS);
                        X = - (lu_JTJ.solve(RHS));
                        T_di = T_di * Sophus::SE3::exp(X).matrix();
                        std::cout << "SQRE " << lss.sqErr/lss.obs << "[" << lss.obs << "] with T_di: " << Sophus::SE3(T_di).log().transpose() << std::endl;
                    }
                //}
            }

        }

        glcamera.SetPose(T_wl.matrix());

        Gpu::BoundingBox bbox_work(T_wl.matrix3x4(), image_w, image_h, K.fu, K.fv, K.u0, K.v0, knear,kfar);
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
        if( show_level < keyframes.size() ) {
            const KinectRgbdKeyframe& kfrm = keyframes[ show_level ];
            adImage.SetImage( kfrm.img_rgb );
            // normalize depth image
            float fMaxDepth = 10.0;
            dDepth.CopyFrom( kfrm.img_d );
            nppiDivC_32f_C1IR( fMaxDepth, dDepth.ptr, dDepth.pitch, dDepth.Size() );
            adDepth.SetImage( dDepth );
        }
        addebug.SetImage( lss_debug );
        adrayimg.SetLevel(viewonly? 0 : show_level);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);
        pangolin::FinishGlutFrame();
    }
}

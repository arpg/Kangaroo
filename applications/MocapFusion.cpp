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
#include "common/ViconTracker.h"
#include "common/PoseGraph.h"
#include "common/GLPoseGraph.h"
#include "common/Handler3dGpuDepth.h"
#include "common/SavePPM.h"
#include "common/PoseGraph.h"
#include "common/GLPoseGraph.h"

#include <CVars/CVar.h>
#include "common/CVarHelpers.h"

#include "MarchingCubes.h"

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>
#include <kangaroo/BoundedVolume.h>
#include "RPG/Utils/GetPot"

using namespace std;
using namespace pangolin;

//Gpu::Mat<ImageKeyframe<Timg>,N> keyframes;

struct KinectKeyframe
{
    KinectKeyframe(int w, int h, Sophus::SE3 T_iw)
        : img(w,h), T_iw(T_iw)
    {
    }

    Sophus::SE3 T_iw;
    Gpu::Image<uchar3, Gpu::TargetDevice, Gpu::Manage> img;
};

int main( int argc, char* argv[] )
{
    GetPot cl( argc, argv );
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
    const Gpu::ImageIntrinsics K(
        camera.GetProperty<double>("Depth0FocalLength", 570.342),
        camera.GetProperty<double>("Depth0FocalLength", 570.342),
        w/2.0 - 0.5, h/2.0 - 0.5
    );
    const double knear = 0.4;
    const double kfar = 4;
    const int volres = 384; //256;
    //const int volres = 256;
    const float volrad = 3;

    const Eigen::Vector4d its(1,2,3,4);

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
//    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(volres,volres,volres,make_float3(-volrad,-volrad,-volrad), make_float3(volrad,volrad,volrad)); // in middle
    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(volres,volres,volres,make_float3(-volrad,-volrad,0.5), make_float3(volrad,volrad,0.5+2*volrad)); // in front
//    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(volres,volres,volres,make_float3(-0.25,-0.5,0.75), make_float3(0.25,0.5,1.25)); // dress form.

    const float3 voxsize = vol.VoxelSizeUnits();

    boost::ptr_vector<KinectKeyframe> keyframes;
    Gpu::Mat<Gpu::ImageKeyframe<uchar3>,10> kfs;

    SceneGraph::GLSceneGraph glgraph;
    SceneGraph::GLAxis glsdf(2.0);
    SceneGraph::GLAxisAlignedBox glboxfrustum;
    SceneGraph::GLAxisAlignedBox glboxvol;
    SceneGraph::GLGrid glGrid(5,1,true);
    SceneGraph::GLAxis glcamera(1.0);
    SceneGraph::GLAxis glcamera_vic(0.5);

    glgraph.AddChild(&glGrid);
    glgraph.AddChild(&glsdf);

    glsdf.AddChild(&glcamera_vic);
    glsdf.AddChild(&glcamera);
    glsdf.AddChild(&glboxvol);
    glsdf.AddChild(&glboxfrustum);

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,K.fu,K.fv,K.u0,K.v0,0.1,1000),
//        pangolin::ModelViewLookAtRDF(0,0,-2,0,0,0,0,-1,0)
        pangolin::ModelViewLookAtRDF(0,5,5,0,0,0,0,0,1)
    );

    Var<bool> run("ui.run", true, true);

    Var<bool> viewonly("ui.view_only", false, true);
    Var<bool> fuse("ui.fuse", true, true);
    Var<bool> reset("ui.reset", true, false);

    Var<int> show_level("ui.Show_Level", 2, 0, MaxLevels-1);

    Var<int> biwin("ui.size",5, 1, 20);
    Var<float> bigs("ui.gs",5, 1E-3, 5);
    Var<float> bigr("ui.gr",100, 1E-3, 200);

    Var<bool> pose_refinement("ui.Pose_Refinement", true, true);
    Var<float> icp_c("ui.icp_c",0.1, 1E-3, 1);

    Var<float> trunc_dist("ui.trunc_dist", 2*length(voxsize), 2*length(voxsize),0.5);
    Var<float> max_w("ui.max_w", 10, 1E-4, 10);
    Var<float> mincostheta("ui.min_cos_theta", 0.1, 0, 1);

    Var<bool> save_kf("ui.Save_KF", false, false);
    Var<float> rgb_fl("ui.RGB_focal_length", 535.7,400,600);
    Var<float> max_rmse("ui.Max_RMSE",0.10,0,0.5);
    Var<float> rmse("ui.RMSE",0);
    Var<bool> add_constraints("ui.Add_Constraints", false, true);
    Var<bool> use_vicon_for_sdf("ui.Use_Vicon_For_SDF", false, true);
    Var<bool> reset_sdf("ui.reset_sdf", false, false);


    ActivateDrawPyramid<float,MaxLevels> adrayimg(ray_i, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawPyramid<float4,MaxLevels> adraycolor(ray_c, GL_RGBA32F, true, true);
//    ActivateDrawPyramid<float4,MaxLevels> adraynorm(ray_n, GL_RGBA32F, true, true);
//    ActivateDrawPyramid<float,MaxLevels> addepth( kin_d, GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawPyramid<float4,MaxLevels> adnormals( ray_n, GL_RGBA32F_ARB, false, true);
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
//    container[4].SetDrawFunction(boost::ref(adraynorm));

    Sophus::SE3 T_sdf_kin;
    bool tracking_good = true;

    pangolin::RegisterKeyPressCallback('l', [&vol,&viewonly]() {LoadPXM("save.vol", vol); viewonly = true;} );
    pangolin::RegisterKeyPressCallback('s', [&vol]() {SavePXM("save.vol", vol); Gpu::SaveMesh("mesh", vol); } );

    PoseGraph posegraph;
    int coord_vicon = posegraph.AddSecondaryCoordinateFrame();
    int kf_sdf = posegraph.AddKeyframe();
    ViconConnection connection("192.168.10.1");
    ViconTracking vicon("Local2", connection);

    Sophus::SE3 T_vicon_ref;
    std::string sRef     = cl.follow( "", 1, "-ref" );
    if(sRef.empty() == false){
        std::string word;
        std::stringstream stream(sRef);
        std::vector<double> vals;
        while( getline(stream, word, ',') ){
            vals.push_back(boost::lexical_cast<double>(word));
        }
        if(vals.size() != 16){
            std::cout << "Attempted to read in reference plane position, but incorrect number of matrix entries provided. Must be 16 numbers" << std::endl;
        }else{
            Eigen::Matrix4d mat;
            for(int ii = 0 ; ii < 4 ; ii++){
                for(int jj = 0 ; jj < 4 ; jj++){
                    mat(ii,jj) = vals[ii*4 + jj];
                }
            }
            T_vicon_ref = mat;
            std::cout << "Ref plane matrix successfully read" << std::endl;
        }
    }

     //set the offset to bring the vicon into the ref coordinate frame
     vicon.SetOffset(T_vicon_ref.inverse());


    GLPoseGraph glposegraph(posegraph);
    glgraph.AddChild(&glposegraph);

    CVarUtils::CreateGetCVar("T_kin_vicon", Sophus::SE3() );
    CVarUtils::AttachCVar("WorkspaceMin", &(vicon.WorkspaceMin()));
    CVarUtils::AttachCVar("WorkspaceMax", &(vicon.WorkspaceMax()));
    CVarUtils::Load("cvars.xml");
    posegraph.GetSecondaryCoordinateFrame(coord_vicon).SetT_wk(CVarUtils::GetCVar<Sophus::SE3>("T_kin_vicon" ) );

    pangolin::RegisterKeyPressCallback(' ', [&add_constraints,&posegraph]() {add_constraints=false; posegraph.Start();} );

    for(long frame=-1; !pangolin::ShouldQuit();)
    {
        const bool go = frame==-1 || run;

        if(rgb_fl.GuiChanged()) {
            save_kf = true;
        }

        if(Pushed(save_kf)) {
            KinectKeyframe* kf = new KinectKeyframe(w,h,T_cd * T_sdf_kin.inverse());
            kf->img.CopyFrom(Gpu::Image<uchar3, Gpu::TargetHost>((uchar3*)img[0].Image.data,w,h));
            keyframes.push_back(kf);
        }

        if(Pushed(reset_sdf)) {
            const Sophus::SE3 T_kin_fiducials = posegraph.GetSecondaryCoordinateFrame(coord_vicon).GetT_wk();
            CVarUtils::SetCVar("T_kin_vicon", T_kin_fiducials);

            // Reset posegraph
            keyframes.clear();
            posegraph.Clear();
            coord_vicon = posegraph.AddSecondaryCoordinateFrame();
            posegraph.GetSecondaryCoordinateFrame(coord_vicon).SetT_wk(T_kin_fiducials);
            kf_sdf = posegraph.AddKeyframe();

//            CVarUtils::AttachCVar<Sophus::SE3>("T_room_sdf", &(posegraph.GetKeyframe(kf_sdf).GetT_wk()) );

            // TODO: Use memory more appropriately.

            vol.bbox.Min() = Gpu::ToCuda(vicon.WorkspaceMin());
            vol.bbox.Max() = Gpu::ToCuda(vicon.WorkspaceMax());
            vol.bbox.Enlarge(make_float3(1.5));
            trunc_dist = 2*length(voxsize);
            Gpu::SdfReset(vol, trunc_dist);

            const Sophus::SE3 T_vicon_fiducials = vicon.T_wf();
            const Sophus::SE3 T_vicon_kin = T_vicon_fiducials * T_kin_fiducials.inverse();
            T_sdf_kin = T_vicon_kin;
            Gpu::SdfFuse(vol, kin_d[0], kin_n[0], T_sdf_kin.inverse().matrix3x4(), K, trunc_dist, max_w, mincostheta );
        }

        const bool newViconData = vicon.IsNewData();
        const Sophus::SE3 T_vicon_figucials = vicon.T_wf();
        const Sophus::SE3 T_vicon_sdf = posegraph.GetKeyframe(kf_sdf).GetT_wk();
        const Sophus::SE3 Tv_sdf_kin = T_vicon_sdf.inverse() * T_vicon_figucials * posegraph.GetSecondaryCoordinateFrame(coord_vicon).GetT_wk().inverse();
        if(use_vicon_for_sdf) {
            tracking_good = newViconData;
            T_sdf_kin = Tv_sdf_kin;
        }

        if(go) {
            camera.Capture(img);

            dKinect.CopyFrom(Gpu::Image<unsigned short, Gpu::TargetHost>((unsigned short*)img[1].Image.data,w,h));
            Gpu::BilateralFilter<float,unsigned short>(kin_d[0],dKinect,bigs,bigr,biwin,200);
            Gpu::ElementwiseScaleBias<float,float,float>(kin_d[0], kin_d[0], 1.0f/1000.0f);

            Gpu::BoxReduceIgnoreInvalid<float,MaxLevels,float>(kin_d);
            for(int l=0; l<MaxLevels; ++l) {
                Gpu::DepthToVbo(kin_v[l], kin_d[l], K[l] );
                Gpu::NormalsFromVbo(kin_n[l], kin_v[l]);
            }

            frame++;
        }

        if(Pushed(reset)) {
            T_sdf_kin = Sophus::SE3();
            Gpu::SdfReset(vol, trunc_dist);
            keyframes.clear();
            posegraph.Clear();
            coord_vicon = posegraph.AddSecondaryCoordinateFrame();
            kf_sdf = posegraph.AddKeyframe();

            // Fuse first kinect frame in.
            Gpu::SdfFuse(vol, kin_d[0], kin_n[0], T_sdf_kin.inverse().matrix3x4(), K, trunc_dist, max_w, mincostheta );
        }

        if(viewonly) {
            Sophus::SE3 T_wv(s_cam.GetModelViewMatrix().Inverse());
            Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( Gpu::BoundingBox(T_wv.matrix3x4(), w, h, K, knear,20) );
            if(work_vol.IsValid()) {
                Gpu::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_wv.matrix3x4(), K, 0.1, 20, trunc_dist, true );
                Gpu::RaycastPlane(ray_d[0], ray_i[0], T_wv.matrix3x4(), K, make_float3(0,0,10000));

                if(keyframes.size() > 0) {
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
                    Gpu::TextureDepth<float4,uchar3,10>(ray_c[0], kfs, ray_d[0], ray_n[0], ray_i[0], T_wv.matrix3x4(), K);
                }
            }
        }else{
            tracking_good = true;

            Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( Gpu::BoundingBox(T_sdf_kin.matrix3x4(), w, h, K, knear,kfar) );
            if(work_vol.IsValid()) {
//                Gpu::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_wl.matrix3x4(), fu, fv, u0, v0, knear, kfar, true );
//                Gpu::BoxReduceIgnoreInvalid<float,MaxLevels,float>(ray_d);
                for(int l=0; l<MaxLevels; ++l) {
                    const Gpu::ImageIntrinsics Kl = K[l];
                    Gpu::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, T_sdf_kin.matrix3x4(), Kl, knear,kfar, true );
                    Gpu::DepthToVbo(ray_v[l], ray_d[l], Kl );
//                    Gpu::DepthToVbo(ray_v[l], ray_d[l], Kl );
//                    Gpu::NormalsFromVbo(ray_n[l], ray_v[l]);
                }

                if( pose_refinement && frame > 0) {
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
                                kin_v[l], ray_v[l], ray_n[l], mKT_lp, mT_pl, icp_c, dScratch, dDebug.SubImage(0,0,w>>l,h>>l)
                            );

                            Eigen::Matrix<double,6,6> sysJTJ = lss.JTJ;
                            Eigen::Matrix<double,6,1> sysJTy = lss.JTy;

                            // Add a week prior on our pose
                            const double motionSigma = use_vicon_for_sdf ? 0.01 : 0.2;
                            const double depthSigma = 0.1;
                            sysJTJ += (depthSigma / motionSigma) * Eigen::Matrix<double,6,6>::Identity();

                            rmse = sqrt(lss.sqErr / lss.obs);
                            tracking_good = rmse < max_rmse;

                            if(l == MaxLevels-1 /*|| use_vicon_for_sdf*/) {
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

                    T_sdf_kin = T_sdf_kin * T_lp.inverse();

                    if(!use_vicon_for_sdf && add_constraints && newViconData && tracking_good) {
                        // Add to pose graph
                        const Sophus::SE3 T_room_sdf = posegraph.GetKeyframe(kf_sdf).GetT_wk();
                        const int kf_kin = posegraph.AddKeyframe(new Keyframe(T_room_sdf * T_sdf_kin));
                        posegraph.AddIndirectUnaryEdge(kf_kin,coord_vicon,T_vicon_figucials);
                        posegraph.AddBinaryEdge(kf_sdf,kf_kin,T_sdf_kin);
                    }
                }
            }

            if(use_vicon_for_sdf && !newViconData) tracking_good = false;

            if( (pose_refinement || use_vicon_for_sdf) && fuse) {
                if(tracking_good) {
                    Gpu::BoundedVolume<Gpu::SDF_t> work_vol = vol.SubBoundingVolume( Gpu::BoundingBox(T_sdf_kin.matrix3x4(), w, h, K, knear,kfar) );
                    if(work_vol.IsValid()) {
                        Gpu::SdfFuse(work_vol, kin_d[0], kin_n[0], T_sdf_kin.inverse().matrix3x4(), K, trunc_dist, max_w, mincostheta );
                    }
                }
            }
        }

//        glAxisVicon.SetPose(T_room_vicon.matrix());
        glcamera_vic.SetPose(Tv_sdf_kin.matrix());
        glsdf.SetPose(T_vicon_sdf.matrix());
        glcamera.SetPose(T_sdf_kin.matrix());

        Gpu::BoundingBox bbox_work(T_sdf_kin.matrix3x4(), w, h, K, knear,kfar);
        bbox_work.Intersect(vol.bbox);

        glboxvol.SetBounds(Gpu::ToEigen(vol.bbox.Min()), Gpu::ToEigen(vol.bbox.Max()) );
        glboxfrustum.SetBounds(Gpu::ToEigen(bbox_work.Min()), Gpu::ToEigen(bbox_work.Max()) );


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

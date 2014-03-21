#include <Eigen/Eigen>
#include <sophus/se3.hpp>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/glvbo.h>

#include <SceneGraph/SceneGraph.h>

#include <kangaroo/kangaroo.h>
#include <kangaroo/BoundedVolume.h>
#include <kangaroo/MarchingCubes.h>
#include <kangaroo/extra/ImageSelect.h>
#include <kangaroo/extra/BaseDisplayCuda.h>
#include <kangaroo/extra/DisplayUtils.h>
#include <kangaroo/extra/Handler3dGpuDepth.h>
#include <kangaroo/extra/SavePPM.h>
#include <kangaroo/extra/SaveMeshlab.h>
#include <kangaroo/extra/CVarHelpers.h>

using namespace std;
using namespace pangolin;

template<typename T>
bool isFinite(const T& x)
{
  return x==x;
}

int main( int argc, char* argv[] )
{
    // Initialise window
    View& container = SetupPangoGLWithCuda(1024, 768);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

    // Open video device
    pangolin::VideoInput video(argc >= 2 ? argv[1] : "openni:[img1=depth]//" );
    const bool use_colour = video.Streams().size() == 2;
    
    std::unique_ptr<unsigned char> vid_buffer( new unsigned char[video.SizeBytes()] );

    std::vector<pangolin::Image<unsigned char> > imgs;    
    const int w = video.Width();
    const int h = video.Height();
    const int MaxLevels = 4;

    const double baseline_m = 0.08; //camera.GetDeviceProperty(<double>("Depth0Baseline", 0) / 100;
    const double depth_focal = 570.342; //camera.GetProperty<double>("Depth0FocalLength", 570.342);
    const roo::ImageIntrinsics K(depth_focal,depth_focal, w/2.0 - 0.5, h/2.0 - 0.5 );
    
    const double knear = 0.4;
    const double kfar = 4;
    const int volres = 256;
    const float volrad = 1.0;

    roo::BoundingBox reset_bb(make_float3(-volrad,-volrad,0.5), make_float3(volrad,volrad,0.5+2*volrad));
//    roo::BoundingBox reset_bb(make_float3(-volrad,-volrad,-volrad), make_float3(volrad,volrad,volrad));

    CVarUtils::AttachCVar<roo::BoundingBox>("BoundingBox", &reset_bb);

    const Eigen::Vector4d its(1,0,2,3);

    // Camera (rgb) to depth
    Eigen::Vector3d c_d(baseline_m,0,0);
    Sophus::SE3d T_cd = Sophus::SE3d(Sophus::SO3d(),c_d).inverse();

    roo::Image<unsigned short, roo::TargetDevice, roo::Manage> dKinect(w,h);
    roo::Image<uchar3, roo::TargetDevice, roo::Manage> drgb(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage> dKinectMeters(w,h);
    roo::Pyramid<float, MaxLevels, roo::TargetDevice, roo::Manage> kin_d(w,h);
    roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> kin_v(w,h);
    roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> kin_n(w,h);
    roo::Image<float4, roo::TargetDevice, roo::Manage>  dDebug(w,h);
    roo::Image<unsigned char, roo::TargetDevice, roo::Manage> dScratch(w*sizeof(roo::LeastSquaresSystem<float,12>),h);

    roo::Pyramid<float, MaxLevels, roo::TargetDevice, roo::Manage> ray_i(w,h);
    roo::Pyramid<float, MaxLevels, roo::TargetDevice, roo::Manage> ray_d(w,h);
    roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> ray_n(w,h);
    roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> ray_v(w,h);
    roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> ray_c(w,h);
    roo::BoundedVolume<roo::SDF_t, roo::TargetDevice, roo::Manage> vol(volres,volres,volres,reset_bb);
    roo::BoundedVolume<float, roo::TargetDevice, roo::Manage> colorVol(volres,volres,volres,reset_bb);

    std::vector<std::unique_ptr<KinectKeyframe> > keyframes;
    roo::Mat<roo::ImageKeyframe<uchar3>,10> kfs;

    SceneGraph::GLSceneGraph glgraph;
    SceneGraph::GLAxis glcamera(0.1);
    SceneGraph::GLAxisAlignedBox glboxfrustum;
    SceneGraph::GLAxisAlignedBox glboxvol;

    glboxvol.SetBounds(roo::ToEigen(vol.bbox.Min()), roo::ToEigen(vol.bbox.Max()) );
    glgraph.AddChild(&glcamera);
    glgraph.AddChild(&glboxvol);
    glgraph.AddChild(&glboxfrustum);

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,K.fu,K.fv,K.u0,K.v0,0.1,1000),
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

    ActivateDrawPyramid<float,MaxLevels> adrayimg(ray_i, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawPyramid<float4,MaxLevels> adraycolor(ray_c, GL_RGBA32F, true, true);
    ActivateDrawPyramid<float4,MaxLevels> adraynorm(ray_n, GL_RGBA32F, true, true);
//    ActivateDrawPyramid<float,MaxLevels> addepth( kin_d, GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawPyramid<float4,MaxLevels> adnormals( kin_n, GL_RGBA32F_ARB, false, true);
    ActivateDrawImage<float4> addebug( dDebug, GL_RGBA32F_ARB, false, true);

    Handler3DGpuDepth rayhandler(ray_d[0], s_cam, AxisNone);
    SetupContainer(container, 4, (float)w/h);
    container[0].SetDrawFunction(std::ref(adrayimg))
                .SetHandler(&rayhandler);
    container[1].SetDrawFunction(SceneGraph::ActivateDrawFunctor(glgraph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );
    container[2].SetDrawFunction(std::ref(adraycolor))
                .SetHandler(&rayhandler);
    container[3].SetDrawFunction(std::ref(adnormals));

    Sophus::SE3d T_wl;

    pangolin::RegisterKeyPressCallback(' ', [&reset,&viewonly]() { reset = true; viewonly=false;} );
    pangolin::RegisterKeyPressCallback('l', [&vol,&viewonly]() {LoadPXM("save.vol", vol); viewonly = true;} );
//    pangolin::RegisterKeyPressCallback('s', [&vol,&colorVol,&keyframes,&rgb_fl,w,h]() {SavePXM("save.vol", vol); SaveMeshlab(vol,keyframes,rgb_fl,rgb_fl,w/2,h/2); } );
//    pangolin::RegisterKeyPressCallback('s', [&vol,&colorVol]() {roo::SaveMesh("mesh",vol,colorVol); } );
    pangolin::RegisterKeyPressCallback('s', [&vol]() {SavePXM("save.vol", vol); } );

    for(long frame=-1; !pangolin::ShouldQuit();)
    {
        const bool go = !viewonly && (frame==-1 || run);

        if(Pushed(save_kf)) {
            KinectKeyframe* kf = new KinectKeyframe(w,h,T_cd * T_wl.inverse());
            kf->img.CopyFrom(roo::Image<uchar3, roo::TargetHost>((uchar3*)imgs[0].ptr,imgs[0].w,imgs[0].h,imgs[0].pitch));
            keyframes.push_back( std::unique_ptr<KinectKeyframe>(kf) );
        }

        if(go) {
            if(video.Grab(vid_buffer.get(), imgs,true,false))
            {
                dKinect.CopyFrom(roo::Image<unsigned short, roo::TargetHost>((unsigned short*)imgs[0].ptr, imgs[0].w, imgs[0].h, imgs[0].pitch ));
                if(use_colour) {
                    drgb.CopyFrom(roo::Image<uchar3, roo::TargetHost>((uchar3*)imgs[1].ptr, imgs[1].w, imgs[1].h, imgs[1].pitch ));
                }
                roo::ElementwiseScaleBias<float,unsigned short,float>(dKinectMeters, dKinect, 1.0f/1000.0f);
                roo::BilateralFilter<float,float>(kin_d[0],dKinectMeters,bigs,bigr,biwin,0.2);

                roo::BoxReduceIgnoreInvalid<float,MaxLevels,float>(kin_d);
                for(int l=0; l<MaxLevels; ++l) {
                    roo::DepthToVbo(kin_v[l], kin_d[l], K[l] );
                    roo::NormalsFromVbo(kin_n[l], kin_v[l]);
                }
    
                frame++;
            }
        }

        if(Pushed(reset)) {
            T_wl = Sophus::SE3d();

            vol.bbox = reset_bb;
            roo::SdfReset(vol);
            keyframes.clear();

            colorVol.bbox = reset_bb;
            roo::SdfReset(colorVol);

            // Fuse first kinect frame in.
            const float trunc_dist = trunc_dist_factor*length(vol.VoxelSizeUnits());
            if(use_colour) {
                roo::SdfFuse(vol, colorVol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), K, drgb, (T_cd * T_wl.inverse()).matrix3x4(), roo::ImageIntrinsics(rgb_fl, drgb), trunc_dist, max_w, mincostheta );
            }else{
                roo::SdfFuse(vol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), K, trunc_dist, max_w, mincostheta );
            }
        }

        if(viewonly) {
            const float trunc_dist = trunc_dist_factor*length(vol.VoxelSizeUnits());

            Sophus::SE3d T_vw(s_cam.GetModelViewMatrix());
            const roo::BoundingBox roi(T_vw.inverse().matrix3x4(), w, h, K, 0, 50);
            roo::BoundedVolume<roo::SDF_t> work_vol = vol.SubBoundingVolume( roi );
            roo::BoundedVolume<float> work_colorVol = colorVol.SubBoundingVolume( roi );
            if(work_vol.IsValid()) {
                if(showcolor) {
                    roo::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, work_colorVol, T_vw.inverse().matrix3x4(), K, 0.1, 50, trunc_dist, true );
                }else{
                    roo::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_vw.inverse().matrix3x4(), K, 0.1, 50, trunc_dist, true );
                }

                if(keyframes.size() > 0) {
                    // populate kfs
                    for( int k=0; k< kfs.Rows(); k++)
                    {
                        if(k < keyframes.size()) {
                            kfs[k].img = keyframes[k]->img;
                            kfs[k].T_iw = keyframes[k]->T_iw.matrix3x4();
                            kfs[k].K = roo::ImageIntrinsics(rgb_fl, kfs[k].img);
                        }else{
                            kfs[k].img.ptr = 0;
                        }
                    }
                    roo::TextureDepth<float4,uchar3,10>(ray_c[0], kfs, ray_d[0], ray_n[0], ray_i[0], T_vw.inverse().matrix3x4(), K);
                }
            }
        }else{
            bool tracking_good = true;

            const roo::BoundingBox roi(roo::BoundingBox(T_wl.matrix3x4(), w, h, K, knear,kfar));
            roo::BoundedVolume<roo::SDF_t> work_vol = vol.SubBoundingVolume( roi );
            if(work_vol.IsValid()) {
//                roo::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_wl.matrix3x4(), fu, fv, u0, v0, knear, kfar, true );
//                roo::BoxReduceIgnoreInvalid<float,MaxLevels,float>(ray_d);
                for(int l=0; l<MaxLevels; ++l) {
                    if(its[l] > 0) {
                        const roo::ImageIntrinsics Kl = K[l];
                        if(showcolor) {
                            roo::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, colorVol, T_wl.matrix3x4(), Kl, knear,kfar, true );
                        }else{
                            roo::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, T_wl.matrix3x4(), Kl, knear,kfar, true );
                        }
                        roo::DepthToVbo(ray_v[l], ray_d[l], Kl );
    //                    roo::DepthToVbo(ray_v[l], ray_d[l], Kl.fu, Kl.fv, Kl.u0, Kl.v0 );
    //                    roo::NormalsFromVbo(ray_n[l], ray_v[l]);
                    }
                }

                if(pose_refinement && frame > 0) {
                    Sophus::SE3d T_lp;

//                    const int l = show_level;
//                    roo::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, T_wl.matrix3x4(), fu/(1<<l), fv/(1<<l), w/(2 * 1<<l) - 0.5, h/(2 * 1<<l) - 0.5, knear,kfar, true );
//                    roo::DepthToVbo(ray_v[l], ray_d[l], fu/(1<<l), fv/(1<<l), w/(2.0f * (1<<l)) - 0.5, h/(2.0f * (1<<l)) - 0.5 );

                    for(int l=MaxLevels-1; l >=0; --l)
                    {
                        const Eigen::Matrix3d Kdepth = K[l].Matrix();

                        for(int i=0; i<its[l]; ++i ) {
                            const Eigen::Matrix<double, 3,4> mKT_lp = Kdepth * T_lp.matrix3x4();
                            const Eigen::Matrix<double, 3,4> mT_pl = T_lp.inverse().matrix3x4();
                            roo::LeastSquaresSystem<float,6> lss = roo::PoseRefinementProjectiveIcpPointPlane(
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
                                T_lp = T_lp * Sophus::SE3d(Sophus::SO3d::exp(x), Eigen::Vector3d(0,0,0) );
                            }else{
                                Eigen::FullPivLU<Eigen::Matrix<double,6,6> > lu_JTJ( sysJTJ );
                                Eigen::Matrix<double,6,1> x = -1.0 * lu_JTJ.solve( sysJTy );
                                if( isFinite(x) ) {
                                    T_lp = T_lp * Sophus::SE3d::exp(x);
                                }
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
                    const roo::BoundingBox roi(T_wl.matrix3x4(), w, h, K, knear,kfar);
                    roo::BoundedVolume<roo::SDF_t> work_vol = vol.SubBoundingVolume( roi );
                    roo::BoundedVolume<float> work_colorVol = colorVol.SubBoundingVolume( roi );
                    if(work_vol.IsValid()) {
                        const float trunc_dist = trunc_dist_factor*length(vol.VoxelSizeUnits());
//                        roo::SdfFuse(work_vol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), K, trunc_dist, max_w, mincostheta );
                        roo::SdfFuse(work_vol, work_colorVol, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), K, drgb, (T_cd * T_wl.inverse()).matrix3x4(), roo::ImageIntrinsics(rgb_fl, drgb), trunc_dist, max_w, mincostheta );
                    }
                }
            }
        }

        glcamera.SetPose(T_wl.matrix());

        roo::BoundingBox bbox_work(T_wl.matrix3x4(), w, h, K.fu, K.fv, K.u0, K.v0, knear,kfar);
        bbox_work.Intersect(vol.bbox);
        glboxfrustum.SetBounds(roo::ToEigen(bbox_work.Min()), roo::ToEigen(bbox_work.Max()) );

//        {
//            CudaScopedMappedPtr var(cbo);
//            roo::Image<uchar4> dCbo((uchar4*)*var,w,h);
//            roo::ConvertImage<uchar4,float4>(dCbo,kin_n[0]);
//        }

//        {
//            CudaScopedMappedPtr var(vbo);
//            roo::Image<float4> dVbo((float4*)*var,w,h);
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
        pangolin::FinishFrame();
    }
}

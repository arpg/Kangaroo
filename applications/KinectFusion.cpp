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

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>

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

    const double fu = camera.GetProperty<double>("Depth0FocalLength", 570.342);
    const double fv = fu;
    const double u0 = w/2.0;
    const double v0 = h/2.0;
    const int volres = 256;

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
    Gpu::Volume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(volres,volres,volres);

    const float3 boxmax = make_float3(1,1,2.5);
    const float3 boxmin = make_float3(-1,-1,0.5);
//    const float3 boxmax = make_float3(0.2,0.2,0.8);
//    const float3 boxmin = make_float3(-0.2,-0.2,0.4);
    const float3 boxsize = boxmax - boxmin;
    const float3 voxsize = boxsize / make_float3(vol.w, vol.h, vol.d);

//    GlBufferCudaPtr vbo(GlArrayBuffer, w,h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
//    GlBufferCudaPtr cbo(GlArrayBuffer, w,h,GL_UNSIGNED_BYTE,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
//    GlBufferCudaPtr ibo(GlElementArrayBuffer, w,h,GL_UNSIGNED_INT,2 );
//    {
//        CudaScopedMappedPtr var(ibo);
//        Gpu::Image<uint2> dIbo((uint2*)*var,w,h);
//        Gpu::GenerateTriangleStripIndexBuffer(dIbo);
//    }

    SceneGraph::GLSceneGraph glgraph;
    SceneGraph::GLAxis glcamera(0.1);
//    SceneGraph::GLVbo glvbo(&vbo,&ibo,&cbo);
    SceneGraph::GLAxisAlignedBox glbox;
    glbox.SetBounds(boxmin.x, boxmin.y, boxmin.z, boxmax.x, boxmax.y, boxmax.z);
    glgraph.AddChild(&glcamera);
//    glcamera.AddChild(&glvbo);
    glgraph.AddChild(&glbox);


    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        ModelViewLookAt(0,0,-2,0,0,0,0,-1,0)
    );

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", true, true);
    Var<bool> lockToCam("ui.Lock to cam", false, true);
    Var<int> show_level("ui.Show Level", 2, 0, MaxLevels-1);
    Var<float> scale("ui.scale",0.0001, 0, 0.001);

    Var<int> biwin("ui.size",5, 1, 20);
    Var<float> bigs("ui.gs",5, 1E-3, 5);
    Var<float> bigr("ui.gr",700, 1E-3, 200);

    Var<bool> pose_refinement("ui.Pose Refinement", false, true);
    Var<bool> reset("ui.reset", false, false);
    Var<float> icp_c("ui.icp c",0.5, 1E-3, 1);
    Var<int> pose_its("ui.pose_its", 5, 0, 10);

    Var<bool> fuse("ui.fuse", false, true);
    Var<bool> fuseonce("ui.fuse once", false, false);

    Var<float> trunc_dist("ui.trunc dist", 2*length(voxsize), 2*length(voxsize),0.5);
    Var<float> max_w("ui.max w", 10, 1E-4, 10);
    Var<float> mincostheta("ui.min cos theta", 0.1, 0, 1);

    pangolin::RegisterKeyPressCallback('l', [&lockToCam](){lockToCam = !lockToCam;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    ActivateDrawPyramid<float,MaxLevels> adrayimg(ray_i, GL_LUMINANCE32F_ARB, true, true);
//    ActivateDrawPyramid<float4,MaxLevels> adraynorm(ray_n, GL_RGBA32F, true, true);
//    ActivateDrawPyramid<float,MaxLevels> addepth( kin_d, GL_LUMINANCE32F_ARB, false, true);
//    ActivateDrawPyramid<float4,MaxLevels> adnormals( kin_n, GL_RGBA32F_ARB, false, true);
    ActivateDrawImage<float4> addebug( dDebug, GL_RGBA32F_ARB, false, true);

    SetupContainer(container, 2, (float)w/h);
//    container[0].SetDrawFunction(boost::ref(addebug));
//    container[2].SetDrawFunction(boost::ref(adnormals));
    container[0].SetDrawFunction(SceneGraph::ActivateDrawFunctor(glgraph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );
    container[1].SetDrawFunction(boost::ref(adrayimg));
//    container[5].SetDrawFunction(boost::ref(adraynorm));

    Sophus::SE3 T_wl;

//    pangolin::RegisterKeyPressCallback(' ', [&posegraph]() {posegraph.Start();} );

    for(long frame=-1; !pangolin::ShouldQuit();)
    {
        const bool go = frame==-1 || run || Pushed(step);

        if(go) {
            camera.Capture(img);
            frame++;

            const int nid = 0;
            dKinect.CopyFrom(Gpu::Image<unsigned short, Gpu::TargetHost>((unsigned short*)img[nid].Image.data,w,h));
            Gpu::BilateralFilter<float,unsigned short>(kin_d[0],dKinect,bigs,bigr,biwin,200);
            Gpu::ElementwiseScaleBias<float,float,float>(kin_d[0], kin_d[0], 1.0f/1000.0f);
            Gpu::BoxReduceIgnoreInvalid<float,MaxLevels,float>(kin_d);
            for(int l=0; l<MaxLevels; ++l) {
                Gpu::DepthToVbo(kin_v[l], kin_d[l], fu/(1<<l), fv/(1<<l), w/(2 * 1<<l), h/(2 * 1<<l) );
                Gpu::NormalsFromVbo(kin_n[l], kin_v[l]);
            }
        }

        if(Pushed(reset) || frame==0) {
            T_wl = Sophus::SE3();
            Gpu::SdfReset(vol, trunc_dist);

            // Fuse first kinect frame in.
            Gpu::SdfFuse(vol, boxmin, boxmax, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), fu, fv, u0, v0, trunc_dist, max_w, mincostheta );
        }

//        // Raycast current view
//        for(int l=MaxLevels-1; l >=0; --l)
//        {
//            Gpu::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], vol, boxmin, boxmax, T_wl.matrix3x4(), fu/(1<<l), fv/(1<<l), w/(2 * 1<<l), h/(2 * 1<<l), 0.2, 8, trunc_dist, true );
//            Gpu::DepthToVbo(ray_v[l], ray_d[l], fu/(1<<l), fv/(1<<l), w/(2 * 1<<l), h/(2 * 1<<l) );
//        }

        if(pose_refinement && frame > 0) {
            const int l = show_level;
            Gpu::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], vol, boxmin, boxmax, T_wl.matrix3x4(), fu/(1<<l), fv/(1<<l), w/(2 * 1<<l), h/(2 * 1<<l), 0.2, 8, trunc_dist, true );
            Gpu::DepthToVbo(ray_v[l], ray_d[l], fu/(1<<l), fv/(1<<l), w/(2 * 1<<l), h/(2 * 1<<l) );

            Sophus::SE3 T_lp;

//            for(int l=MaxLevels-1; l >=0; --l)
            {
                const int l = show_level;
                const int lits = pose_its;
                Eigen::Matrix3d Kdepth;
                Kdepth << fu/(1<<l), 0, w/(2 * 1<<l),   0, fu/(1<<l), h/(2 * 1<<l),  0,0,1;

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

        if(Pushed(fuseonce) || fuse) {
            // integrate gtd into TSDF
            Gpu::SdfFuse(vol, boxmin, boxmax, kin_d[0], kin_n[0], T_wl.inverse().matrix3x4(), fu, fv, u0, v0, trunc_dist, max_w, mincostheta );
        }

        glcamera.SetPose(T_wl.matrix());

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
        adrayimg.SetLevel(show_level);
//        adraynorm.SetLevel(show_level);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);
        pangolin::FinishGlutFrame();
    }
}

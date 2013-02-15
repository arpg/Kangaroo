#include <Eigen/Eigen>
#include <sophus/se3.hpp>

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
using namespace Gpu;

void Save(std::string prefix, Sophus::SE3d T_vc, float f)
{
    const Eigen::Matrix<double,6,1> cartT_vc = SceneGraph::GLT2Cart(T_vc.matrix());
    cout << cartT_vc << endl;

    ofstream ft(prefix + "_f_T_vc.txt");
    ft << f << endl;
    ft << cartT_vc;
    ft.close();
}

int main( int argc, char* argv[] )
{
    // RDF transforms
    Eigen::Matrix3d RDFvision;RDFvision<< 1,0,0,  0,1,0,  0,0,1;
    Eigen::Matrix3d RDFrobot; RDFrobot << 0,1,0,  0,0, 1,  1,0,0;
    Eigen::Matrix3d RDFvicon; RDFvicon << 0,-1,0,  0,0, -1,  1,0,0;
    Eigen::Matrix4d T_vis_ro = Eigen::Matrix4d::Identity();
    T_vis_ro.block<3,3>(0,0) = RDFvision.transpose() * RDFrobot;
    Eigen::Matrix4d T_ro_vis = Eigen::Matrix4d::Identity();
    T_ro_vis.block<3,3>(0,0) = RDFrobot.transpose() * RDFvision;
    Eigen::Matrix4d T_vis_vic = Eigen::Matrix4d::Identity();
    T_vis_vic.block<3,3>(0,0) = RDFvision.transpose() * RDFvicon;
    Eigen::Matrix4d T_vic_vis = Eigen::Matrix4d::Identity();
    T_vic_vis.block<3,3>(0,0) = RDFvicon.transpose() * RDFvision;

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

    const double dfl = camera.GetProperty<double>("DepthFocalLength", 570.342);

//    const double ifl = dfl;
//    const double baseline = -camera.GetProperty<double>("RGBDepthBaseline", 80) / 1000.0;
//    Eigen::Matrix3d Krgb;
//    Krgb << ifl, 0, w/2.0,   0, ifl, h/2.0,  0,0,1;
//    const Sophus::SE3d T_cd = Sophus::SE3d(Sophus::SO3d(),Eigen::Vector3d(baseline,0,0)).inverse();
//    Eigen::Matrix<double,3,4> KT_cd = Krgb * T_cd.matrix3x4();

    Image<unsigned short, TargetDevice, Manage> dKinect(w,h);
    Image<uchar3, TargetDevice, Manage>  imgRGB(w,h);
    Image<unsigned char, TargetDevice, Manage>  imgI(w,h);

    Pyramid<float, MaxLevels, TargetDevice, Manage> pyrD(w,h);
    Pyramid<float4, MaxLevels, TargetDevice, Manage> pyrV(w,h);
    Pyramid<float4, MaxLevels, TargetDevice, Manage> pyrN(w,h);
    Pyramid<float4, MaxLevels, TargetDevice, Manage> pyrVprev(w,h);
    Image<float4, TargetDevice, Manage>  dDebug(w,h);
    Image<unsigned char, TargetDevice,Manage> dScratch(w*sizeof(LeastSquaresSystem<float,12>),h);


    GlBufferCudaPtr vbo(GlArrayBuffer, w*h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBufferCudaPtr cbo(GlArrayBuffer, w*h,GL_UNSIGNED_BYTE,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBuffer ibo = pangolin::MakeTriangleStripIboForVbo(w,h);

    SceneGraph::GLSceneGraph glgraph;
    SceneGraph::GLAxis glcamera(0.1);
    SceneGraph::GLVbo glvbo(&vbo,&ibo,&cbo);
    SceneGraph::GLAxis glvicon;
    SceneGraph::GLGrid glGrid(10,1,false);
    glgraph.AddChild(&glcamera);
    glcamera.AddChild(&glvbo);
    glgraph.AddChild(&glvicon);
    glgraph.AddChild(&glGrid);

    PoseGraph posegraph;
    GLPoseGraph glposegraph(posegraph);
    glgraph.AddChild(&glposegraph);

    int coord_z = posegraph.AddSecondaryCoordinateFrame( Sophus::SE3d(T_vic_vis) );

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        ModelViewLookAt(0,5,5,0,0,0,0,0,1)
    );

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", true, true);
    Var<bool> lockToCam("ui.Lock to cam", false, true);
    Var<int> show_level("ui.Show Level", 1, 0, MaxLevels-1);
    Var<float> scale("ui.scale",0.0001, 0, 0.001);

    Var<int> biwin("ui.size",10, 1, 20);
    Var<float> bigs("ui.gs",10, 1E-3, 5);
    Var<float> bigr("ui.gr",700, 1E-3, 200);

    Var<bool> pose_refinement("ui.Pose Refinement", false, true);
    Var<bool> reset("ui.reset", false, false);
    Var<float> icp_c("ui.icp c",0.5, 1E-3, 1);
    Var<int> pose_its("ui.pose_its", 10, 0, 10);

    pangolin::RegisterKeyPressCallback('l', [&lockToCam](){lockToCam = !lockToCam;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    ActivateDrawImage<unsigned char> adrgb(imgI, GL_LUMINANCE8, false, true);
//    ActivateDrawImage<float> addepth( imgD, GL_LUMINANCE32F_ARB, false, true);
//    ActivateDrawImage<float4> adnormals( imgN, GL_RGBA32F_ARB, false, true);
    ActivateDrawPyramid<float,MaxLevels> addepth( pyrD, GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawPyramid<float4,MaxLevels> adnormals( pyrN, GL_RGBA32F_ARB, false, true);
    ActivateDrawImage<float4> addebug( dDebug, GL_RGBA32F_ARB, false, true);

    SetupContainer(container, 4, (float)w/h);
    container[0].SetDrawFunction(boost::ref(addebug));
    container[1].SetDrawFunction(boost::ref(addepth));
    container[2].SetDrawFunction(boost::ref(adnormals));
    container[3].SetDrawFunction(SceneGraph::ActivateDrawFunctor(glgraph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );

    Sophus::SE3d T_wl;

    GetPot cl(argc,argv);
    const int nid = cl.follow(0, "-camid");
    const std::string cam_name = "Local" + boost::lexical_cast<std::string>(nid);
    ViconTracking vicon(cam_name, "192.168.10.1");

    pangolin::RegisterKeyPressCallback(' ', [&posegraph]() {posegraph.Start();} );
    pangolin::RegisterKeyPressCallback('s', [cam_name,&posegraph,coord_z,dfl]() {Save(cam_name, posegraph.GetSecondaryCoordinateFrame(coord_z).GetT_wk(), dfl);});

    for(unsigned long frame=0; !pangolin::ShouldQuit();)
    {
        const bool go = frame==0 || run || Pushed(step);

        if(go) {
            camera.Capture(img);
            const bool newViconData = vicon.IsNewData();
            const Sophus::SE3d T_wv = vicon.T_wf();

            // Save current as last
            pyrVprev.Swap(pyrV);

//            imgRGB.CopyFrom(Image<uchar3, TargetHost>((uchar3*)img[0].Image.data,w,h));
//            Gpu::ConvertImage<unsigned char, uchar3>(imgI, imgRGB);
            dKinect.CopyFrom(Image<unsigned short, TargetHost>((unsigned short*)img[nid].Image.data,w,h));
            BilateralFilter<float,unsigned short>(pyrD[0],dKinect,bigs,bigr,biwin,200);
            BoxReduceIgnoreInvalid<float,MaxLevels,float>(pyrD);
            for(int l=0; l<MaxLevels; ++l) {
                DepthToVbo(pyrV[l], pyrD[l], dfl/(1<<l), dfl/(1<<l), w/(2 * 1<<l), h/(2 * 1<<l), 1.0f/1000.0f );
                NormalsFromVbo(pyrN[l], pyrV[l]);
            }

            if( Pushed(reset) || frame == 0 ) {
                T_wl = Sophus::SE3d();
                if(vicon.IsConnected()) {
                    T_wl = vicon.T_wf();
                }
            }

            if(pose_refinement) {
                Sophus::SE3d T_pl;

                if(frame > 0) {
    //                for(int l=MaxLevels-1; l >=0; --l)
                    {
                        const int l = show_level;
                        const int lits = pose_its;
                        Eigen::Matrix3d Kdepth;
                        Kdepth << dfl/(1<<l), 0, w/(2 * 1<<l),   0, dfl/(1<<l), h/(2 * 1<<l),  0,0,1;

                        for(int i=0; i<lits; ++i ) {
                            const Eigen::Matrix<double, 3,4> mKT_pl = Kdepth * T_pl.matrix3x4();
                            const Eigen::Matrix<double, 3,4> mT_lp = T_pl.inverse().matrix3x4();
                            Gpu::LeastSquaresSystem<float,6> lss = PoseRefinementProjectiveIcpPointPlane(
                                        pyrVprev[l], pyrV[l], pyrN[l], mKT_pl, mT_lp, icp_c, dScratch, dDebug.SubImage(0,0,w>>l,h>>l)
                            );
                            Eigen::FullPivLU<Eigen::Matrix<double,6,6> > lu_JTJ( (Eigen::Matrix<double,6,6>)lss.JTJ );
                            Eigen::Matrix<double,6,1> x = -1.0 * lu_JTJ.solve( (Eigen::Matrix<double,6,1>)lss.JTy );
                            T_pl = T_pl * Sophus::SE3d::exp(x);
                        }
                    }

                    T_wl = T_wl * T_pl;
                }

                {
                    // Add to pose graph
                    Keyframe* kf = new Keyframe(T_wv);
                    const int kfid = posegraph.AddKeyframe(kf);
                    if(newViconData) {
                        posegraph.AddUnaryEdge(kfid, T_wv);
                    }
                    if(kfid > 0 ) {
                        posegraph.AddIndirectBinaryEdge(kfid-1, kfid, coord_z, T_pl );
                    }
                }
            }

//            glcamera.SetPose(T_wl.matrix());
            glcamera.SetPose(T_wv.matrix());
            glvbo.SetPose(posegraph.GetSecondaryCoordinateFrame(coord_z).GetT_wk().matrix());
//            glvicon.SetPose(T_wv.matrix());

//            if(frame==0 | Pushed(save_ref))
            {
                CudaScopedMappedPtr var(cbo);
                Gpu::Image<uchar4> dCbo((uchar4*)*var,w,h);
                Gpu::ConvertImage<uchar4,float4>(dCbo,pyrN[0]);
//                ColourVbo(dCbo, pyrV[0], imgRGB, KT_cd);
            }


            {
                CudaScopedMappedPtr var(vbo);
                Gpu::Image<float4> dVbo((float4*)*var,w,h);
                dVbo.CopyFrom(pyrV[0]);
            }

            frame++;
        }

        /////////////////////////////////////////////////////////////
        // Draw
        addebug.SetImage(dDebug.SubImage(0,0,w>>show_level,h>>show_level));
        addepth.SetImageScale(scale);
        addepth.SetLevel(show_level);
        adnormals.SetLevel(show_level);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);
        pangolin::FinishGlutFrame();
    }

//    Save(strnid, posegraph.GetSecondaryCoordinateFrame(coord_z).GetT_wk(), dfl);

    posegraph.Stop();
}

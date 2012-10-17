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

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;

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

    const double dfl = camera.GetProperty<double>("DepthFocalLength", 570.342);
    const double ifl = dfl;
    const double baseline = -camera.GetProperty<double>("RGBDepthBaseline", 80) / 1000.0;

    Eigen::Matrix3d Krgb;
    Krgb << ifl, 0, w/2.0,   0, ifl, h/2.0,  0,0,1;
    const Sophus::SE3 T_cd = Sophus::SE3(Sophus::SO3(),Eigen::Vector3d(baseline,0,0)).inverse();
    Eigen::Matrix<double,3,4> KT_cd = Krgb * T_cd.matrix3x4();

    Image<unsigned short, TargetDevice, Manage> dKinect(w,h);
    Image<uchar3, TargetDevice, Manage>  imgRGB(w,h);
    Image<unsigned char, TargetDevice, Manage>  imgI(w,h);

    Pyramid<float, MaxLevels, TargetDevice, Manage> pyrD(w,h);
    Pyramid<float4, MaxLevels, TargetDevice, Manage> pyrV(w,h);
    Pyramid<float4, MaxLevels, TargetDevice, Manage> pyrN(w,h);
    Pyramid<float4, MaxLevels, TargetDevice, Manage> pyrVprev(w,h);
    Image<float4, TargetDevice, Manage>  dDebug(w,h);
    Image<unsigned char, TargetDevice,Manage> dScratch(w*sizeof(LeastSquaresSystem<float,12>),h);


    GlBufferCudaPtr vbo(GlArrayBuffer, w,h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBufferCudaPtr cbo(GlArrayBuffer, w,h,GL_UNSIGNED_BYTE,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBufferCudaPtr ibo(GlElementArrayBuffer, w,h,GL_UNSIGNED_INT,2 );
    {
        CudaScopedMappedPtr var(ibo);
        Gpu::Image<uint2> dIbo((uint2*)*var,w,h);
        GenerateTriangleStripIndexBuffer(dIbo);
    }

    SceneGraph::GLSceneGraph graph;
    SceneGraph::GLAxis glaxis;
    graph.AddChild(&glaxis);
    SceneGraph::GLAxis glcamera(0.1);
    SceneGraph::GLVbo glvbo(&vbo,&ibo,&cbo);
    graph.AddChild(&glcamera);
    glcamera.AddChild(&glvbo);

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,dfl,dfl, w/2, h/2,1E-2,1E3),
        ModelViewLookAtRDF(0,0,0,0,0,1,0,-1,0)
    );

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", true, true);
    Var<bool> lockToCam("ui.Lock to cam", false, true);
    Var<int> show_level("ui.Show Level", 1, 0, MaxLevels-1);
    Var<float> scale("ui.scale",0.0001, 0, 0.001);

    Var<int> biwin("ui.size",10, 1, 20);
    Var<float> bigs("ui.gs",10, 1E-3, 5);
    Var<float> bigr("ui.gr",700, 1E-3, 200);

    Var<bool> pose_refinement("ui.Pose Refinement", true, true);
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
    container[3].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );

    Sophus::SE3 T_wl;

//    ViconTracking vicon("Beef0", "192.168.1.10");
//    ofstream file_vicon("vicon.txt");
    ofstream file_icp("icp.txt");

    for(unsigned long frame=0; !pangolin::ShouldQuit();)
    {
        const bool go = frame==0 || run || Pushed(step);

        if(go) {
            camera.Capture(img);

            // Save current as last
            pyrVprev.Swap(pyrV);

//            imgRGB.CopyFrom(Image<uchar3, TargetHost>((uchar3*)img[0].Image.data,w,h));
            Gpu::ConvertImage<unsigned char, uchar3>(imgI, imgRGB);
            dKinect.CopyFrom(Image<unsigned short, TargetHost>((unsigned short*)img[1].Image.data,w,h));
            BilateralFilter<float,unsigned short>(pyrD[0],dKinect,bigs,bigr,biwin,200);
            BoxReduceIgnoreInvalid<float,MaxLevels,float>(pyrD);
            for(int l=0; l<MaxLevels; ++l) {
                DepthToVbo(pyrV[l], pyrD[l], dfl/(1<<l), dfl/(1<<l), w/(2 * 1<<l), h/(2 * 1<<l), 1.0f/1000.0f );
                NormalsFromVbo(pyrN[l], pyrV[l]);
            }

            if( Pushed(reset) ) {
                T_wl = Sophus::SE3();
            }

            if(pose_refinement) {
                if(frame > 0) {
                    Sophus::SE3 T_pl;
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
                            T_pl = T_pl * Sophus::SE3::exp(x);
                        }
                    }

                    T_wl = T_wl * T_pl;
                    glcamera.SetPose(T_wl.matrix());

                    file_icp << SceneGraph::GLT2Cart(T_pl.matrix()).transpose() << endl;
                }else{
                    file_icp << SceneGraph::GLT2Cart(Sophus::SE3().matrix()).transpose() << endl;
                }
            }

//            if(frame==0 | Pushed(save_ref))
            {
                CudaScopedMappedPtr var(cbo);
                Gpu::Image<uchar4> dCbo((uchar4*)*var,w,h);
                ColourVbo(dCbo, pyrV[0], imgRGB, KT_cd);
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
}

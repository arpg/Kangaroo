#include <Eigen/Eigen>
#include <Sophus/se3.h>

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

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;

int main( int /*argc*/, char* argv[] )
{
    // Initialise window
    View& container = SetupPangoGLWithCuda(1024, 768);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

    // Open video device
    CameraDevice camera = OpenRpgCamera("Kinect://");
    std::vector<rpg::ImageWrapper> img;
    camera.Capture(img);
    const int w = img[0].width();
    const int h = img[0].height();

    const double dfl = camera.GetProperty<double>("DepthFocalLength", 0);
    const double ifl = dfl;
    const double baseline = camera.GetProperty<double>("RGBDepthBaseline", 0) / 1000.0;

    cout << ifl << endl;
    cout << dfl << endl;
    cout << baseline << endl;

    Eigen::Matrix3d Krgb;
    Krgb << ifl, 0, w/2.0,
            0, ifl, h/2.0,
            0,0,1;

    const Sophus::SE3 T_cd = Sophus::SE3(Sophus::SO3(),Eigen::Vector3d(baseline,0,0)).inverse();

    Image<unsigned short, TargetDevice, Manage> dKinect(w,h);
    Image<uchar3, TargetDevice, Manage>  imgRGB(w,h);
    Image<unsigned char, TargetDevice, Manage>  imgI(w,h);
    Image<float, TargetDevice, Manage>  imgD(w,h);
    Image<float, TargetDevice, Manage>  imgDfilt(w,h);
    Image<float4, TargetDevice, Manage> imgV(w,h);
    Image<float4, TargetDevice, Manage> imgN(w,h);

    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgg(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgu(w,h);
    Gpu::Image<float2, Gpu::TargetDevice, Gpu::Manage> imgp(w,h);
    Image<float2, Gpu::TargetDevice, Gpu::Manage> imgv(w,h);
    Image<float4, Gpu::TargetDevice, Gpu::Manage> imgq(w,h);
    Image<float, Gpu::TargetDevice, Gpu::Manage> imgr(w,h);


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
    SceneGraph::GLVbo glvbo(&vbo,&ibo,&cbo);
    graph.AddChild(&glvbo);

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,dfl,dfl, w/2, h/2,1E-2,1E3),
        ModelViewLookAtRDF(0,0,0,0,0,1,0,-1,0)
    );

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", true, true);
    Var<bool> lockToCam("ui.Lock to cam", false, true);

    Var<float> scale("ui.scale",0.0001, 0, 0.001);

    Var<bool> save_ref("ui.Save Reference", true, false);

    Var<bool> tgv_do("ui.tgv", false, true);
    Var<float> sigma("ui.sigma", 0.002, 0, 0.01);
    Var<float> tau("ui.tau", 0.002, 0, 0.01);
    Var<float> tgv_a1("ui.alpha1", 100, 0, 0.5);
    Var<float> tgv_k("ui.k", 2, 1, 10);
    Var<float> tgv_delta("ui.delta", 0.1, 0, 0.2);


    pangolin::RegisterKeyPressCallback('l', [&lockToCam](){lockToCam = !lockToCam;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    ActivateDrawImage<unsigned char> adrgb(imgI, GL_LUMINANCE8, false, true);
    ActivateDrawImage<float> addepth( imgD, GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float4> adnormals( imgN, GL_RGBA32F_ARB, false, true);
    ActivateDrawImage<float> addenoised( imgu, GL_LUMINANCE32F_ARB, false, true);

    SetupContainer(container, 4, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adrgb));
    container[1].SetDrawFunction(boost::ref(addepth));
    container[2].SetDrawFunction(boost::ref(adnormals));
    container[3].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );

    for(unsigned long frame=0; !pangolin::ShouldQuit();)
    {
        const bool go = frame==0 || run || Pushed(step);

        if(go) {
            if(camera.Capture(img)) {
                imgRGB.CopyFrom(Image<uchar3, TargetHost>((uchar3*)img[0].Image.data,w,h));
                Gpu::ConvertImage<unsigned char, uchar3>(imgI, imgRGB);
                dKinect.CopyFrom(Image<unsigned short, TargetHost>((unsigned short*)img[1].Image.data,w,h));
//                FilterBadKinectData(imgD,dKinect);
                ConvertImage<float,unsigned short>(imgD, dKinect);
                frame++;
            }

            if(frame==0 | Pushed(save_ref)) {
                imgg.CopyFrom(imgD);
                imgu.CopyFrom(imgg);
                imgp.Memset(0);
                imgq.Memset(0);
                imgr.Memset(0);
                Gpu::GradU(imgv,imgu);

                CudaScopedMappedPtr var(cbo);
                Gpu::Image<uchar4> dCbo((uchar4*)*var,w,h);
                Eigen::Matrix<double,3,4> KT_cd = Krgb * T_cd.matrix3x4();
                ColourVbo(dCbo, imgV, imgRGB, KT_cd);
            }

            if(tgv_do) {
                for(int i=0; i<20; ++i ) {
                    const float tgv_a0 = tgv_k * tgv_a1;
                    Gpu::TGV_L1_DenoisingIteration(imgu,imgv,imgp,imgq,imgr,imgg,tgv_a0, tgv_a1, sigma, tau, tgv_delta);
                }
            }

            FilterBadKinectData(imgDfilt,imgu);
            DepthToVbo(imgV, imgDfilt, dfl, dfl, w/2, h/2, 1.0f/1000.0f );
            NormalsFromVbo(imgN, imgV);

            {
                CudaScopedMappedPtr var(vbo);
                Gpu::Image<float4> dVbo((float4*)*var,w,h);
                dVbo.CopyFrom(imgV);
            }

        }

        /////////////////////////////////////////////////////////////
        // Draw
        addepth.SetImageScale(scale);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);
        pangolin::FinishGlutFrame();
    }
}

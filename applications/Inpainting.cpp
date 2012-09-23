#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <npp.h>

#include <fiducials/drawing.h>

#include "common/RpgCameraOpen.h"
#include "common/DisplayUtils.h"
#include "common/BaseDisplayCuda.h"
#include "common/ImageSelect.h"

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

using namespace std;
using namespace pangolin;

int main( int argc, char* argv[] )
{
    // Open video device
    CameraDevice video = OpenRpgCamera(argc,argv,1);

    // Capture first image
    std::vector<rpg::ImageWrapper> images;
    video.Capture(images);

    // Image dimensions
    const unsigned int w = images[0].width();
    const unsigned int h = images[0].height();

    // Initialise window
    View& container = SetupPangoGLWithCuda(180+2*w, h,180);

    // Allocate Camera Images on device for processing
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> img(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgg(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgu(w,h);
    Gpu::Image<float2, Gpu::TargetDevice, Gpu::Manage> imgp(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgdivp(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imglambda(w,h);

    const bool bilinear = false;
    ActivateDrawImage<float> adg(imgg, GL_LUMINANCE32F_ARB, bilinear, true);
    ActivateDrawImage<float> adu(imgu, GL_LUMINANCE32F_ARB, bilinear, true);
    ActivateDrawImage<float> addivp(imgdivp, GL_LUMINANCE32F_ARB, bilinear, true);
    ActivateDrawImage<float> adlambda(imglambda, GL_LUMINANCE32F_ARB, bilinear, true);

    Handler2dImageSelect handler2d(w,h);
    SetupContainer(container, 4, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adg)).SetHandler(&handler2d);
    container[1].SetDrawFunction(boost::ref(adu)).SetHandler(&handler2d);
    container[2].SetDrawFunction(boost::ref(addivp)).SetHandler(&handler2d);
    container[3].SetDrawFunction(boost::ref(adlambda)).SetHandler(&handler2d);

    Var<bool> run("ui.run", true, true);
    Var<bool> step("ui.step", false, false);

    const float L = sqrt(8);
    Var<float> sigma("ui.sigma", 1.0f/L, 0, 0.1);
    Var<float> tau("ui.tau", 1.0f/L, 0, 0.1);
    Var<float> lambda("ui.lambda", 1.2, 0, 10);
    Var<float> alpha("ui.alpha", 0.002, 0, 0.005);

    Var<float> r("ui.r", 10, 1, 50);

    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        bool go = (frame==0) || Pushed(step);

        if(go) {
            video.Capture(images);
            img.MemcpyFromHost(images[0].Image.data );
            Gpu::ElementwiseScaleBias<float,unsigned char,float>(imgg, img, 1.0f/255.0f);
            imgu.CopyFrom(imgg);
            imgp.Memset(0);
            imgdivp.Memset(0);
            Gpu::Fill<float>(imglambda, 1.0);
        }

        go |= run;
        if(go) {
            for(int i=0; i<10; ++i ) {
                Gpu::HuberGradU_DualAscentP(imgp,imgu,sigma,alpha);
                Gpu::Divergence(imgdivp,imgp);
                Gpu::L2_u_minus_g_PrimalDescent(imgu,imgp,imgg,imglambda, tau, lambda);
            }
        }

        if(handler2d.IsSelected()) {
            Eigen::Vector2d p = handler2d.GetSelectedPoint(true);
            Gpu::PaintCircle<float>(imglambda, 0.0f, p[0], p[1], r);
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

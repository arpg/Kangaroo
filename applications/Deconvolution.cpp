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
    CameraDevice video = OpenRpgCamera(argc,argv,2);

    // Capture first image
    std::vector<rpg::ImageWrapper> images;
    video.Capture(images);

    // Image dimensions
    const unsigned int w = images[0].width();
    const unsigned int h = images[0].height();
    const unsigned int kw = images[1].width();
    const unsigned int kh = images[1].height();

    // Initialise window
    View& container = SetupPangoGLWithCuda(180+2*w, h,180);
    SetupContainer(container, 4, (float)w/h);

    // Allocate Camera Images on device for processing
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> img(w,h);
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> kernel(kw,kh);

    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imggt(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgg(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgk(kw,kh);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgkT(kw,kh);

    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage>  imgu(w,h);
    Gpu::Image<float2, Gpu::TargetDevice, Gpu::Manage> imgp(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage>  imgq(w,h);

    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage>  imgAu(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage>  imgATq(w,h);

    ActivateDrawImage<float> adgt(imggt, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawImage<float> adg(imgg, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawImage<float> adk(imgk, GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adu(imgu, GL_LUMINANCE32F_ARB, true, true);

    Handler2dImageSelect handler2d(w,h);
    container[0].SetDrawFunction(boost::ref(adgt)).SetHandler(&handler2d);
    container[1].SetDrawFunction(boost::ref(adk)).SetHandler(&handler2d);
    container[2].SetDrawFunction(boost::ref(adg)).SetHandler(&handler2d);
    container[3].SetDrawFunction(boost::ref(adu)).SetHandler(&handler2d);

    Var<bool> nextImage("ui.step", false, false);
    Var<bool> go("ui.go", false, true);

    Var<float> sigma_q("ui.sigma q", 0.001, 0, 0.01);
    Var<float> sigma_p("ui.sigma p", 0.001, 0, 0.01);
    Var<float> tau("ui.tau", 0.001, 0, 0.01);
    Var<float> lambda("ui.lambda", 1000, 0, 100);
//    Var<float> alpha("ui.alpha", 0.002, 0, 0.005);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        const bool reset = (frame==0) || Pushed(nextImage);

        if(reset) {
            video.Capture(images);
            img.MemcpyFromHost(images[0].Image.data );
            kernel.MemcpyFromHost(images[1].Image.data );
            Gpu::ElementwiseScaleBias<float,unsigned char,float>(imggt, img, 1.0f/255.0f);
            Gpu::ElementwiseScaleBias<float,unsigned char,float>(imgk, kernel, 1.0f/255.0f);
            Gpu::Transpose<float,float>(imgkT, imgk);
            Gpu::Convolution<float,float,float,float>(imgg, imggt, imgk, kw/2, kh/2);
            imgu.CopyFrom(imgg);
            imgp.Memset(0);
            imgq.Memset(0);
        }

        if(go) {
            for(int i=0; i<1; ++i ) {
                Gpu::TVL1GradU_DualAscentP(imgp,imgu,sigma_p);
                Gpu::Convolution<float,float,float,float>(imgAu, imgu, imgk, kw/2, kh/2);
                Gpu::DeconvolutionDual_qAscent(imgq,imgAu,imgg,sigma_q,lambda);
                Gpu::Convolution<float,float,float,float>(imgATq, imgq, imgkT, kw/2, kh/2);
                Gpu::Deconvolution_uDescent(imgu,imgp,imgATq, tau, lambda);
            }
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

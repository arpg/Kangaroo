#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <npp.h>

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
    View& container = SetupPangoGLWithCuda(2*w, h);

    // Allocate Camera Images on device for processing
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> img(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgg(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgu(w,h);
    Gpu::Image<float2, Gpu::TargetDevice, Gpu::Manage> imgp(w,h);

    Gpu::Image<float2, Gpu::TargetDevice, Gpu::Manage> imgv(w,h);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> imgq(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> imgr(w,h);

    ActivateDrawImage<float> adg(imgg, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawImage<float> adu(imgu, GL_LUMINANCE32F_ARB, true, true);

    Handler2dImageSelect handler2d(w,h);
    SetupContainer(container, 2, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adg)).SetHandler(&handler2d);
    container[1].SetDrawFunction(boost::ref(adu)).SetHandler(&handler2d);

    Var<bool> nextImage("ui.step", false, false);
    Var<bool> go("ui.go", false, true);

    const float L = sqrt(8);
    Var<float> sigma("ui.sigma", 0.04, 0, 0.1);
    Var<float> tau("ui.tau", 0.05, 0, 0.1);
    Var<float> lambda("ui.lambda", 1.2, 0, 10);
    Var<float> alpha("ui.alpha", 0.002, 0, 0.005);

    Var<bool> tgv_do("ui.tgv", false, true);
    Var<float> tgv_a1("ui.alpha1", 0.9, 0, 0.5);
//    Var<float> tgv_a0("ui.alpha0", 0.5, 0, 0.5);
    Var<float> tgv_k("ui.k", 10, 1, 10);
    Var<float> tgv_delta("ui.delta", 0.1, 0, 0.2);


    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        const bool reset = (frame==0) || Pushed(nextImage);

        if(reset) {
            video.Capture(images);
            img.MemcpyFromHost(images[0].Image.data );
            Gpu::ElementwiseScaleBias<float,unsigned char,float>(imgg, img, 1.0f/255.0f);
            imgu.CopyFrom(imgg);
            imgp.Memset(0);

            imgq.Memset(0);
            imgr.Memset(0);

            Gpu::GradU(imgv,imgu);
        }

        if(go) {
            if(!tgv_do) {
                for(int i=0; i<10; ++i ) {
                    Gpu::HuberGradU_DualAscentP(imgp,imgu,sigma,alpha);
                    Gpu::L2_u_minus_g_PrimalDescent(imgu,imgp,imgg, tau, lambda);
                }
            }else{
                for(int i=0; i<20; ++i ) {
                    const float tgv_a0 = tgv_k * tgv_a1;
                    Gpu::TGV_L1_DenoisingIteration(imgu,imgv,imgp,imgq,imgr,imgg,tgv_a0, tgv_a1, sigma, tau, tgv_delta);
                }
            }
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

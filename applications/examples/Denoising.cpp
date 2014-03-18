#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>

#include <kangaroo/extra/DisplayUtils.h>
#include <kangaroo/extra/BaseDisplayCuda.h>
#include <kangaroo/extra/ImageSelect.h>

using namespace std;
using namespace pangolin;

int main( int argc, char* argv[] )
{
    // Open video device
    const std::string vid_uri = argc >= 2 ? argv[1] : "";    
    pangolin::VideoInput video(vid_uri);
    if(video.PixFormat().format != "GRAY8")
        throw pangolin::VideoException("Wrong format. Gray8 required.");

    // Image dimensions and host copy
    const unsigned int w = video.Width();
    const unsigned int h = video.Height();
    roo::Image<unsigned char, roo::TargetHost, roo::Manage> host(w,h);

    // Initialise window
    View& container = SetupPangoGLWithCuda(2*w, h);

    // Allocate Camera Images on device for processing
    roo::Image<unsigned char, roo::TargetDevice, roo::Manage> img(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage> imgg(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage> imgu(w,h);
    roo::Image<float2, roo::TargetDevice, roo::Manage> imgp(w,h);

    roo::Image<float2, roo::TargetDevice, roo::Manage> imgv(w,h);
    roo::Image<float4, roo::TargetDevice, roo::Manage> imgq(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage> imgr(w,h);

    ActivateDrawImage<float> adg(imgg, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawImage<float> adu(imgu, GL_LUMINANCE32F_ARB, true, true);

    Handler2dImageSelect handler2d(w,h);
    SetupContainer(container, 2, (float)w/h);
    container[0].SetDrawFunction(std::ref(adg)).SetHandler(&handler2d);
    container[1].SetDrawFunction(std::ref(adu)).SetHandler(&handler2d);

    Var<bool> nextImage("ui.step", false, false);
    Var<bool> go("ui.go", false, true);

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
            if(video.GrabNext(host.ptr)) {
                img.CopyFrom(host);
            }
            roo::ElementwiseScaleBias<float,unsigned char,float>(imgg, img, 1.0f/255.0f);
            imgu.CopyFrom(imgg);
            imgp.Memset(0);

            imgq.Memset(0);
            imgr.Memset(0);

            roo::GradU(imgv,imgu);
        }

        if(go) {
            if(!tgv_do) {
                for(int i=0; i<10; ++i ) {
                    roo::HuberGradU_DualAscentP(imgp,imgu,sigma,alpha);
                    roo::L2_u_minus_g_PrimalDescent(imgu,imgp,imgg, tau, lambda);
                }
            }else{
                for(int i=0; i<20; ++i ) {
                    const float tgv_a0 = tgv_k * tgv_a1;
                    roo::TGV_L1_DenoisingIteration(imgu,imgv,imgp,imgq,imgr,imgg,tgv_a0, tgv_a1, sigma, tau, tgv_delta);
                }
            }
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishFrame();
    }
}

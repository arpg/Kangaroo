#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include <kangaroo/extra/DisplayUtils.h>
#include <kangaroo/extra/BaseDisplayCuda.h>
#include <kangaroo/extra/ImageSelect.h>

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>

using namespace std;
using namespace pangolin;

int main( int argc, char* argv[] )
{
    // Open video device
    const std::string vid_uri = argc >= 2 ? argv[1] : "";    
    pangolin::VideoInput video(vid_uri);
    if(video.Streams().size() != 2)
        throw pangolin::VideoException("Requires two video streams.");    
    if(video.PixFormat().format != "GRAY8" || video.Streams()[1].PixFormat().format != "GRAY8" )
        throw pangolin::VideoException("Wrong format. Gray8 required.");    
    unsigned char vid_buffer[video.SizeBytes()];

    // Image dimensions
    const unsigned int w = video.Streams()[0].Width();
    const unsigned int h = video.Streams()[0].Height();
    const unsigned int kw = video.Streams()[1].Width();
    const unsigned int kh = video.Streams()[1].Height();

    // Initialise window
    View& container = SetupPangoGLWithCuda(180+2*w, h,180);

    // Allocate Camera Images on device for processing
    roo::Image<unsigned char, roo::TargetDevice, roo::Manage> img(w,h);
    roo::Image<unsigned char, roo::TargetDevice, roo::Manage> kernel(kw,kh);

    roo::Image<float, roo::TargetDevice, roo::Manage> imggt(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage> imgg(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage> imgk(kw,kh);
    roo::Image<float, roo::TargetDevice, roo::Manage> imgkT(kw,kh);

    roo::Image<float, roo::TargetDevice, roo::Manage>  imgu(w,h);
    roo::Image<float2, roo::TargetDevice, roo::Manage> imgp(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage>  imgq(w,h);

    roo::Image<float, roo::TargetDevice, roo::Manage>  imgAu(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage>  imgATq(w,h);

    ActivateDrawImage<float> adgt(imggt, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawImage<float> adg(imgg, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawImage<float> adk(imgk, GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adu(imgu, GL_LUMINANCE32F_ARB, true, true);
    ActivateDrawImage<float> adAu(imgAu, GL_LUMINANCE32F_ARB, true, true);

    Handler2dImageSelect handler2d(w,h);
    SetupContainer(container, 5, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adgt)).SetHandler(&handler2d);
    container[1].SetDrawFunction(boost::ref(adk)).SetHandler(&handler2d);
    container[2].SetDrawFunction(boost::ref(adg)).SetHandler(&handler2d);
    container[3].SetDrawFunction(boost::ref(adu)).SetHandler(&handler2d);
    container[4].SetDrawFunction(boost::ref(adAu)).SetHandler(&handler2d);

    Var<bool> nextImage("ui.step", false, false);
    Var<bool> go("ui.go", false, true);

    Var<float> sigma_q("ui.sigma q", 0.2, 0, 0.01);
    Var<float> sigma_p("ui.sigma p", 0.2, 0, 0.01);
    Var<float> tau("ui.tau", 0.001, 0, 0.01);
    Var<float> lambda("ui.lambda", 20, 0, 100);
    Var<float> alpha("ui.alpha", 0.0, 0, 0.005);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        const bool reset = (frame==0) || Pushed(nextImage);

        if(reset) {
            std::vector<pangolin::Image<unsigned char> > images;
            if(video.Grab(vid_buffer,images)) {
                img.MemcpyFromHost(images[0].ptr );
                kernel.MemcpyFromHost(images[1].ptr );
                roo::ElementwiseScaleBias<float,unsigned char,float>(imggt, img, 1.0f/255.0f);
                roo::ElementwiseScaleBias<float,unsigned char,float>(imgk, kernel, 1.0f/255.0f);
                roo::Transpose<float,float>(imgkT, imgk);
                roo::Convolution<float,float,float,float>(imgg, imggt, imgk, kw/2, kh/2);
                imgu.CopyFrom(imgg);
                imgp.Memset(0);
                imgq.Memset(0);
            }
        }

        if(go) {
            for(int i=0; i<1; ++i ) {
//                roo::TVL1GradU_DualAscentP(imgp,imgu,sigma_p);
                roo::HuberGradU_DualAscentP(imgp,imgu,sigma_p,alpha);
                roo::Convolution<float,float,float,float>(imgAu, imgu, imgk, kw/2, kh/2);
                roo::DeconvolutionDual_qAscent(imgq,imgAu,imgg,sigma_q,lambda);
                roo::Convolution<float,float,float,float>(imgATq, imgq, imgkT, kw/2, kh/2);
                roo::Deconvolution_uDescent(imgu,imgp,imgATq, tau, lambda);
            }
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishFrame();
    }
}

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include <kangaroo/common/DisplayUtils.h>
#include <kangaroo/common/BaseDisplayCuda.h>
#include <kangaroo/common/ImageSelect.h>

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
                Gpu::ElementwiseScaleBias<float,unsigned char,float>(imggt, img, 1.0f/255.0f);
                Gpu::ElementwiseScaleBias<float,unsigned char,float>(imgk, kernel, 1.0f/255.0f);
                Gpu::Transpose<float,float>(imgkT, imgk);
                Gpu::Convolution<float,float,float,float>(imgg, imggt, imgk, kw/2, kh/2);
                imgu.CopyFrom(imgg);
                imgp.Memset(0);
                imgq.Memset(0);
            }
        }

        if(go) {
            for(int i=0; i<1; ++i ) {
//                Gpu::TVL1GradU_DualAscentP(imgp,imgu,sigma_p);
                Gpu::HuberGradU_DualAscentP(imgp,imgu,sigma_p,alpha);
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

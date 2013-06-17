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
    View& container = SetupPangoGLWithCuda(1024, 768);
    SetupContainer(container, 2, (float)w/h);

    // Texture we will use to display camera images
    GlTextureCudaArray tex8(w,h,GL_LUMINANCE8);
    GlTextureCudaArray texf(w,h,GL_LUMINANCE32F_ARB);

    // Allocate Camera Images on device for processing
    roo::Image<unsigned char, roo::TargetDevice, roo::Manage> dImg(w,h);
    roo::Image<float, roo::TargetDevice, roo::Manage> dImgFilt(w,h);

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", false, true);
    Var<int> bilateralWinSize("ui.size",10, 1, 20);
    Var<float> bigs("ui.gs",5, 1E-3, 5);
    Var<float> bigr("ui.gr",20, 1E-3, 200);

    Var<int> domedits("ui.median its",1, 1, 10);
    Var<bool> domed5x5("ui.median 5x5", true, true);
    Var<bool> domed3x3("ui.median 3x3", false, true);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        const bool go = (frame==0) || run || Pushed(step);

        if(go) {
            if(video.GrabNext(host.ptr)) {
                dImg.CopyFrom(host);                
            }
        }

        if(go || GuiVarHasChanged() ) {
            roo::BilateralFilter<float,unsigned char>(dImgFilt,dImg,bigs,bigr,bilateralWinSize);
//            ConvertImage<float,unsigned char>(dImgFilt,dImg);

            for(int i=0; i < domedits; ++i ) {
                if(domed3x3) {
                    roo::MedianFilter3x3(dImgFilt,dImgFilt);
                }

                if(domed5x5) {
                    roo::MedianFilter5x5(dImgFilt,dImgFilt);
                }
            }

            // normalise dImgFilt
            nppiDivC_32f_C1IR(255,dImgFilt.ptr,dImgFilt.pitch,dImgFilt.Size());
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        if(container[0].IsShown()) {
            container[0].Activate();
            tex8 << dImg;
            tex8.RenderToViewportFlipY();
        }

        if(container[1].IsShown()) {
            container[1].Activate();
            texf << dImgFilt;
            texf.RenderToViewportFlipY();
        }

        pangolin::FinishFrame();
    }
}

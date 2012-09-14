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

#include <kangaroo/kangaroo.h>

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;

int main( int argc, char* argv[] )
{
    // Open video device
    CameraDevice video = OpenRpgCamera(argc,argv,1);

    // Capture first image
    std::vector<rpg::ImageWrapper> img;
    video.Capture(img);

    // Image dimensions
    const unsigned int w = img[0].width();
    const unsigned int h = img[0].height();

    // Initialise window
    View& container = SetupPangoGLWithCuda(1024, 768);
    SetupContainer(container, 2, (float)w/h);

    // Texture we will use to display camera images
    GlTextureCudaArray tex8(w,h,GL_LUMINANCE8);
    GlTextureCudaArray texf(w,h,GL_LUMINANCE32F_ARB);

    // Allocate Camera Images on device for processing
    Image<unsigned char, TargetDevice, Manage> dImg(w,h);
    Image<float, TargetDevice, Manage> dImgFilt(w,h);

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
            video.Capture(img);
            dImg.MemcpyFromHost(img[0].Image.data );
        }

        if(go || GuiVarHasChanged() ) {
            BilateralFilter<float,unsigned char>(dImgFilt,dImg,bigs,bigr,bilateralWinSize);
//            ConvertImage<float,unsigned char>(dImgFilt,dImg);

            for(int i=0; i < domedits; ++i ) {
                if(domed3x3) {
                    MedianFilter3x3(dImgFilt,dImgFilt);
                }

                if(domed5x5) {
                    MedianFilter5x5(dImgFilt,dImgFilt);
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

        pangolin::FinishGlutFrame();
    }
}

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/glsl.h>
#include <npp.h>

#include <fiducials/drawing.h>

#include "common/RpgCameraOpen.h"
#include "common/DisplayUtils.h"
#include "common/BaseDisplay.h"

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

    cout << "Image has dimensions: " << w << "x" << h << endl;

    // Initialise window
    View& container = SetupPangoGLWithCuda(1024, 768);
    SetupContainer(container, 2, (float)w/h);

    // Texture we will use to display camera images
    GlTextureCudaArray tex8(w,h,GL_LUMINANCE8);
    GlTextureCudaArray texf(w,h,GL_LUMINANCE32F_ARB);

    // Allocate Camera Images on device for processing
    Image<unsigned char, TargetDevice, Manage> dImg(w,h);
    Image<int,TargetDevice,Manage> dRowPrefixSum(w,h);
    Image<int,TargetDevice,Manage> dRowPrefixSumT(h,w);
    Image<int,TargetDevice,Manage> dIntegralImage(h,w);
    Image<float,TargetDevice,Manage> dBox(h,w);

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", false, true);
    Var<int> rad("ui.radius",10, 1, 20);

    for(unsigned long frame=0; !pangolin::ShouldQuit() /*&& frame < 100*/; ++frame)
    {
        const bool go = (frame==0) || run || Pushed(step);

        if(go) {
            video.Capture(img);
            dImg.MemcpyFromHost(img[0].Image.data );
        }

        if(go || GuiVarHasChanged() ) {
            PrefixSumRows<int,unsigned char>(dRowPrefixSum, dImg);
            Transpose<int,int>(dRowPrefixSumT,dRowPrefixSum);
            PrefixSumRows<int,int>(dIntegralImage, dRowPrefixSumT);
            BoxFilterIntegralImage<float,int>(dBox,dIntegralImage,rad);
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
            texf << dBox;
            GlSlUtilities::Scale(1/255.0f);
            texf.RenderToViewportFlipY();
            GlSlUtilities::UseNone();
        }

        pangolin::FinishGlutFrame();
    }
}

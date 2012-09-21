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
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> imgs(w,h);
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> scratch(w,h);

    ActivateDrawImage<unsigned char> adg(img, GL_LUMINANCE8, true, true);
    ActivateDrawImage<unsigned char> ads(imgs, GL_LUMINANCE8, true, true);

    Handler2dImageSelect handler2d(w,h);
    SetupContainer(container, 2, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adg)).SetHandler(&handler2d);
    container[1].SetDrawFunction(boost::ref(ads)).SetHandler(&handler2d);

    Var<bool> run("ui.run", true, true);
    Var<bool> step("ui.step", false, false);
    Var<int> thresh("ui.threshold", 10, 0, 255);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        bool go = (frame==0) || run || Pushed(step);

        if(go) {
            video.Capture(images);
            img.MemcpyFromHost(images[0].Image.data );
//            Gpu::ElementwiseScaleBias<float,unsigned char,float>(imgg, img, 1.0f/255.0f);
        }

        go |= GuiVarHasChanged();
        if(go) {
            Gpu::SegmentTest(imgs, img, thresh);
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

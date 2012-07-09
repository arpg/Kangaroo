#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <npp.h>

#include <fiducials/drawing.h>

#include "RpgCameraOpen.h"
#include "DisplayUtils.h"
#include "ScanlineRectify.h"
#include "CudaImage.h"
#include "kernel.h"

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;

int main( int /*argc*/, char* argv[] )
{
    const int UI_WIDTH = 180;

    // Open video device
//    const std::string cam_uri =
    CameraDevice camera = OpenRpgCamera(
//        "AlliedVision:[NumChannels=2,DataSourceDir=/Users/slovegrove/data/AlliedVisionCam,CamUUID0=5004955,CamUUID1=5004954,ImageBinningX=2,ImageBinningY=2,ImageWidth=694,ImageHeight=518]//"
//        "FileReader:[NumChannels=2,DataSourceDir=/Users/slovegrove/data/CityBlock-Noisy,Channel-0=left.*pgm,Channel-1=right.*pgm,StartFrame=0,BufferSize=120]//"
//        "FileReader:[NumChannels=2,DataSourceDir=/Users/slovegrove/data/xb3,Channel-0=left.*pgm,Channel-1=right.*pgm,StartFrame=0,BufferSize=120]//"
//        "FileReader:[NumChannels=2,DataSourceDir=/Users/slovegrove/data/20120515/20090822_212628/rect_images,Channel-0=.*left.pnm,Channel-1=.*right.pnm,StartFrame=500,BufferSize=60]//"
//        "Dvi2Pci:[NumChannels=2,ImageWidth=640,ImageHeight=480,BufferCount=60]//"
        "FileReader:[NumChannels=1,DataSourceDir=/Users/slovegrove/data/noisy,Channel-0=gradient-noise.png,StartFrame=0,BufferSize=120]//"
//        "FileReader:[NumChannels=1,DataSourceDir=/Users/slovegrove/data/noisy,Channel-0=lucy.png,StartFrame=0,BufferSize=120]//"
    );

//    CameraDevice camera = OpenPangoCamera(
//        "file:[stream=0,fmt=GRAY8]///Users/slovegrove/data/3DCam/DSCF0051.AVI",
//        "file:[stream=1,fmt=GRAY8]///Users/slovegrove/data/3DCam/DSCF0051.AVI"
//    );

    // Capture first image
    std::vector<rpg::ImageWrapper> img;
    camera.Capture(img);

    // Check we received one or more images
    if(img.empty()) {
        std::cerr << "Failed to capture first image from camera" << std::endl;
        return -1;
    }

    // N cameras, each w*h in dimension, greyscale
    const unsigned int w = img[0].width();
    const unsigned int h = img[0].height();

    // Setup OpenGL Display (based on GLUT)
    pangolin::CreateGlutWindowAndBind(__FILE__,UI_WIDTH+2*w,h);
    glewInit();

    // Initialise CUDA, allowing it to use OpenGL context
    cudaGLSetGLDevice(0);

    // Setup default OpenGL parameters
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_BLEND);
    glEnable (GL_LINE_SMOOTH);
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    // Tell the base view to arrange its children equally
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, Attach::Pix(UI_WIDTH));

    View& container = CreateDisplay()
            .SetBounds(0,1.0, Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(LayoutEqual);

    const int N = 2;
    for(int i=0; i<N; ++i ) {
        View& disp = CreateDisplay().SetAspect((double)w/h);
        container.AddDisplay(disp);
    }

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
    Var<float> bigo("ui.go",0, 0, 1);

    Var<int> domedits("ui.median its",1, 1, 10);
    Var<bool> domed5x5("ui.median 5x5", true, true);
    Var<bool> domed3x3("ui.median 3x3", false, true);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        const bool go = (frame==0) || run || Pushed(step);

        if(go) {
            camera.Capture(img);
            dImg.MemcpyFromHost(img[0].Image.data );
        }

        if(go || GuiVarHasChanged() ) {
            RobustBilateralFilter(dImgFilt,dImg,bigs,bigr,bigo,bilateralWinSize);
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

        container[0].Activate();
        tex8 << dImg;
        tex8.RenderToViewportFlipY();

        container[1].Activate();
        texf << dImgFilt;
        texf.RenderToViewportFlipY();

        pangolin::RenderViews();
        pangolin::FinishGlutFrame();
    }
}

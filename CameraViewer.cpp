#include <Eigen/Eigen>
#include <pangolin/pangolin.h>

#include "RpgCameraOpen.h"

using namespace std;
using namespace pangolin;

int main( int /*argc*/, char* argv[] )
{
    // Open video device
    const std::string cam_uri =
            "AlliedVision:[NumChannels=2,CamUUID0=5004955,CamUUID1=5004954,ImageBinningX=2,ImageBinningY=2,ImageWidth=694,ImageHeight=518]//";
//            "FileReader:[DataSourceDir=/home/slovegrove/data/CityBlock-Noisy,Channel-0=left.*pgm,Channel-1=right.*pgm,StartFrame=0]//";
//            "Dvi2Pci:[NumImages=2,ImageWidth=640,ImageHeight=480,BufferCount=60]//";

    CameraDevice camera = OpenRpgCamera(cam_uri);

    // Capture first image
    std::vector<rpg::ImageWrapper> img;
    camera.Capture(img);

    // Check we received one or more images
    if(img.empty()) {
        std::cerr << "Failed to capture first image from camera" << std::endl;
        return -1;
    }

    // N cameras, each w*h in dimension, greyscale
    const int N = img.size();
    const int w = img[0].width();
    const int h = img[0].height();

    // Setup OpenGL Display (based on GLUT)
    pangolin::CreateGlutWindowAndBind("Main",N*w,h);

    // Setup default OpenGL parameters
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_BLEND);
    glEnable (GL_LINE_SMOOTH);
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    // Create Smart viewports for each camera image that preserve aspect
    View* view[N];
    for(int i=0; i<N; ++i ) {
        view[i] = &CreateDisplay().SetAspect((double)w/h);
    }

    // Tell the base view to arrange its children equally
    DisplayBase().SetLayout(LayoutEqual);

    // Texture we will use to display camera images
    GlTexture tex(w,h,GL_LUMINANCE8);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        camera.Capture(img);
        for(int i=0; i<N; ++i ) {
            view[i]->Activate();
            tex.Upload(img[i].Image.data,GL_LUMINANCE, GL_UNSIGNED_BYTE);
            tex.RenderToViewportFlipY();
        }

        pangolin::FinishGlutFrame();
    }
}

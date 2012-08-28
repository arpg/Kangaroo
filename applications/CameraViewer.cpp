#include <Eigen/Eigen>
#include <pangolin/pangolin.h>

#include "common/RpgCameraOpen.h"

using namespace std;
using namespace pangolin;

int main( int argc, char* argv[] )
{
    // Open video device
    CameraDevice camera = OpenRpgCamera(argc,argv);

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
    pangolin::CreateGlutWindowAndBind(__FILE__,N*w,h);

    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    GlTexture tex(w,h,GL_LUMINANCE8);

    // Create Smart viewports for each camera image that preserve aspect
    View* view[N];
    for(int i=0; i<N; ++i ) {
        view[i] = &CreateDisplay().SetAspect((double)w/h);
    }

    // Tell the base view to arrange its children equally
    DisplayBase().SetLayout(LayoutEqual);

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

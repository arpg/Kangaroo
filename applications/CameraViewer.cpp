#include <Eigen/Eigen>
#include <pangolin/pangolin.h>
#include <pangolin/timer.h>

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

    cout << N << " Cameras " << endl;
    for(int i=0; i<N; ++i ) {
        cout << "Camera" << i << ": Size: " << img[i].width() << "x" << img[i].height()
             << ", Channels: " << img[i].Image.channels()
             << ", Bytes Per Channel: " << img[i].Image.elemSize1() << endl;
    }

    // Setup OpenGL Display (based on GLUT)
    pangolin::CreateGlutWindowAndBind(__FILE__,N*w,h);

    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    GlTexture tex(w,h,GL_RGB8);

    // Create Smart viewports for each camera image that preserve aspect
    View* view[N];
    for(int i=0; i<N; ++i ) {
        view[i] = &CreateDisplay().SetAspect((double)w/h);
    }

    // Tell the base view to arrange its children equally
    DisplayBase().SetLayout(LayoutEqual);

    bool run = true;
    bool step = false;

    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    pangolin::Timer timer;

    for(unsigned long frame=0; !pangolin::ShouldQuit();)
    {
        const bool go = run || Pushed(step);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        if(go) {
            if(frame>0) {
                camera.Capture(img);
            }
            frame++;

            if(frame%30 == 0) {
                cout << "FPS: " << frame / timer.Elapsed_s() << "\r";
                cout.flush();
            }
        }

        for(int i=0; i<N; ++i ) {
            view[i]->Activate();
            tex.Upload(
                img[i].Image.data,
                img[i].Image.channels() == 1 ? GL_LUMINANCE : GL_RGB,
                img[i].Image.elemSize1() == 1 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT
            );
            tex.RenderToViewportFlipY();
        }

        pangolin::FinishGlutFrame();
    }
}

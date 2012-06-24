#include <pangolin/pangolin.h>

using namespace std;
using namespace pangolin;

int main( int /*argc*/, char* argv[] )
{
    string uri[2] = {
        "file:[stream=0]///home/slovegrove/data/3DCam/DSCF0051.AVI",
        "file:[stream=1]///home/slovegrove/data/3DCam/DSCF0051.AVI"
    };

    VideoInput video[2] = { uri[0], uri[1] };
    VideoPixelFormat vid_fmt = VideoFormatFromString(video[0].PixFormat());

    const unsigned w = video[0].Width();
    const unsigned h = video[1].Height();

    // Create Glut window
    pangolin::CreateGlutWindowAndBind(__FILE__,w,h);

    // Create viewport for video with fixed aspect
    View& screen = DisplayBase();
    screen.SetLayout(LayoutEqual);

    CreateDisplay().SetAspect((float)w/h);
    CreateDisplay().SetAspect((float)w/h);

    // OpenGl Texture for video frame
    GlTexture texVideo(w,h,GL_RGBA8);

    unsigned char* img = new unsigned char[video[0].SizeBytes()];

    for(int frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        glColor3f(1,1,1);

        for(int i=0; i<2; ++i ) {
            screen[i].Activate();
            video[i].GrabNext(img,true);
            texVideo.Upload(img, vid_fmt.channels==1 ? GL_LUMINANCE:GL_RGB, GL_UNSIGNED_BYTE);
            texVideo.RenderToViewportFlipY();
        }

        // Swap back buffer with front and process window events via GLUT
        pangolin::FinishGlutFrame();
        usleep(5000);
    }

    delete[] img;
}

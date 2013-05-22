#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include <kangaroo/kangaroo.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;

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
    const unsigned int w = video.Width();
    const unsigned int h = video.Height();

    // Setup OpenGL Display (based on GLUT)
    pangolin::CreateGlutWindowAndBind(__FILE__,2*w,2*h);

    // Initialise CUDA, allowing it to use OpenGL context
    cudaGLSetGLDevice(0);

    // Setup default OpenGL parameters
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_BLEND);
    glEnable (GL_LINE_SMOOTH);
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    // Tell the base view to arrange its children equally
    View& screen = CreateDisplay().SetAspect((double)w/h);

    // Texture we will use to display camera images
    GlTextureCudaArray texrgb(w,h,GL_RGBA8);

    // Allocate Camera Images on device for processing
    Gpu::Image<unsigned char, TargetDevice, Manage> dCamImg[] = {{w,h},{w,h}};
    Gpu::Image<uchar4, TargetDevice, Manage> d3d(w,h);

    int shift = 0;
    bool run = true;

    pangolin::RegisterKeyPressCallback('0', [&shift](){shift=0;} );
    pangolin::RegisterKeyPressCallback('=', [&shift](){shift++;} );
    pangolin::RegisterKeyPressCallback('-', [&shift](){shift--;} );
    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        if( run || frame == 0 ) {
            std::vector<pangolin::Image<unsigned char> > images;
            if(video.Grab(vid_buffer,images)) {
                // Upload images to device
                for(int i=0; i<2; ++i ) {
                    dCamImg[i].CopyFrom( Gpu::Image<unsigned char,TargetHost>(
                        images[i].ptr, images[i].w,
                        images[i].h, images[i].pitch
                    ));
                }
            }
        }

        // Perform Processing
        MakeAnaglyth(d3d, dCamImg[0], dCamImg[1], shift);

        // Draw Anaglyph
        screen.Activate();
        CopyDevMemtoTex(d3d.ptr, d3d.pitch, texrgb );
        texrgb.RenderToViewportFlipY();

        pangolin::FinishGlutFrame();
        usleep(1000000 / 30);
    }
}

#include <thread>
#include <mutex>

#include <SimpleGui/Gui.h>
#include <SimpleGui/GetPot>
#include <SimpleGui/GLMesh.h>

#include <RPG/Devices/Camera/CameraDevice.h>

using namespace std;
using namespace Eigen;

void glDrawTexturesQuad(float t, float b, float l, float r)
{
    glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2f(l,b);
    glTexCoord2f(1,0); glVertex2f(r,b);
    glTexCoord2f(1,1); glVertex2f(r,t);
    glTexCoord2f(0,1); glVertex2f(l,t);
    glEnd();
}

class Application
{
public:
    Application()
        : window(0, 0, 640*2, 480, __FILE__ )
    {
        Init();
    }

    void Init()
    {
        // -sdir /home/slovegrove/data/CityBlock-Noisy -sf 0
        cam.SetProperty("StartFrame",    0);
        cam.SetProperty("DataSourceDir", "/home/slovegrove/data/CityBlock-Noisy" );
        cam.SetProperty("Channel-0",     "left.*pgm" );
        cam.SetProperty("Channel-1",     "right.*pgm" );
        cam.SetProperty("NumChannels",   2 );
        cam.InitDriver("FileReader");

        // Capture first images to get dimensions
        cam.Capture(img);
        width = img[0].width();
        height = img[0].height();

        // Create two OpenGL textures for stereo images
        glGenTextures(2, m_glTex);

        // Allocate texture memory on GPU
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        window.AddPostRenderCallback( Application::PostRender, this);
    }

    static void PostRender(GLWindow*, void* data)
    {
        Application* self = (Application*)data;
        self->Draw();
    }

    void CameraRun()
    {
        while(1) {
            imageMutex.lock();
            cam.Capture(img);
            imageMutex.unlock();

            usleep(33000);
        }
    }

    int Run()
    {
        // Run Camera Loop
        std::thread camThread( std::bind( &Application::CameraRun, this ) );

        // Run GUI
        return window.Run();
    }

    void Draw()
    {
        if( img.size() >= 2 )
        {
            imageMutex.lock();
            glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_LUMINANCE,GL_UNSIGNED_BYTE,img[0].Image.data);
            glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_LUMINANCE,GL_UNSIGNED_BYTE,img[1].Image.data);
            imageMutex.unlock();

            // Display frames
            glDisable(GL_LIGHTING);
            glDisable(GL_COLOR_MATERIAL );

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glClearColor (0.0, 0.0, 0.0, 0.0);
            glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glEnable(GL_TEXTURE_2D);

            glColor3f (1.0, 1.0, 1.0);

            // Draw right image
            glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
            glDrawTexturesQuad(-1,1,-1,0);

            // Draw left image
            glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
            glDrawTexturesQuad(-1,1,0,1);

            glDisable(GL_TEXTURE_2D);
        }
    }

    std::vector<rpg::ImageWrapper> img;
    GLuint m_glTex[2];
    int width, height;
    std::mutex imageMutex;

    CameraDevice cam;
    GLWindow window;
};

int main (int argc, char** argv){
    Application app;
    return app.Run();
}

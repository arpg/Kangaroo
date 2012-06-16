#include <thread>
#include <mutex>

#include <SimpleGui/Gui.h>
#include <SimpleGui/GetPot>
#include <SimpleGui/GLMesh.h>

#include <RPG/Devices/Camera/CameraDevice.h>

#include <fiducials/tracker.h>
#include <fiducials/drawing.h>

using namespace std;
using namespace Eigen;

const int DESIRED_WIDTH = 640;
const int DESIRED_HEIGHT = 480;

const int WINDOW_WIDTH  = 2 * DESIRED_WIDTH;
const int WINDOW_HEIGHT = 2 * DESIRED_HEIGHT;

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
        : window(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, __FILE__ ),
          shouldQuit(false)
    {
        Init();
    }

    ~Application()
    {
    }

    void Init()
    {
        if(false) {
            // Setup Camera
            camera.SetProperty("StartFrame",    0);
            camera.SetProperty("DataSourceDir", "/home/slovegrove/data/CityBlock-Noisy" );
            camera.SetProperty("Channel-0",     "left.*pgm" );
            camera.SetProperty("Channel-1",     "right.*pgm" );
            camera.SetProperty("NumChannels",   2 );
            camera.InitDriver("FileReader");
            camera.Capture(img);
        }else{
            camera.SetProperty("NumChannels", 2 );
            camera.SetProperty("CamUUID0", 5004955);
            camera.SetProperty("CamUUID1", 5004954);
            camera.SetProperty("ImageWidth", DESIRED_WIDTH);
            camera.SetProperty("ImageHeight", DESIRED_HEIGHT);
            if(!camera.InitDriver( "AlliedVision" )) {
                cerr << "Couldn't start driver for camera " << endl;
                exit(1);
            }
            camera.Capture(img);
        }
        width = img[0].width();
        height = img[0].height();

        // Setup OpenGL Textures
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

        // Setup OpenGL Render Callback
        window.AddPostRenderCallback( Application::PostRender, this);

    }

    static void PostRender(GLWindow*, void* data)
    {
        Application* self = (Application*)data;
        self->Draw();
    }

    void CameraRun()
    {
        // Unit hell!
        const double ppi = 72; // Points Per Inch
        const double USwp = 11 * ppi;
        const double UShp = 8.5 * ppi;
        const double mpi = 0.0254; // meters per inch
        const double mpp = mpi / ppi; // meters per point
        const double unit = 1; //mpp;

        // Setup Trackers
        Tracker tracker[2] = {{width,height},{width,height}};
        for(int i=0; i<2; ++i) {
            tracker[i].target.GenerateRandom(60,unit*USwp*25/(842.0),unit*USwp*75/(842.0),unit*USwp*40/(842.0),Eigen::Vector2d(unit*USwp,unit*UShp));
            tracker[i].target.SaveEPS("stereo.eps");
        }

        Matrix<double,9,1> cam_params; // = Var<Matrix<double,9,1> >("cam_params");
        cam_params << 0.808936, 1.06675, 0.495884, 0.520504, 0.180668, -0.354284, -0.00169838, 0.000600873, 0.0;
        MatlabCamera camParams( width,height, width*cam_params[0],height*cam_params[1], width*cam_params[2],height*cam_params[3], cam_params[4], cam_params[5], cam_params[6], cam_params[7], cam_params[8]);

        while(!shouldQuit) {
//            imageMutex.lock();
            camera.Capture(img);
//            imageMutex.unlock();

            for(int i=0; i<1; ++i) {
                tracker[i].ProcessFrame(camParams,img[i].Image.data);
                T_gw[i] = tracker[i].T_gw;
            }
        }
    }

    int Run()
    {
        // Run Camera Loop
        std::thread camThread( std::bind( &Application::CameraRun, this ) );

        // Run GUI
        window.Run();
        shouldQuit = true;

        camThread.join();
    }

    void Draw()
    {
        // Draw Images
        glViewport(0,DESIRED_HEIGHT,WINDOW_WIDTH,DESIRED_HEIGHT);

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

        // Draw Poses
        glViewport(0,0,DESIRED_WIDTH,DESIRED_HEIGHT);
        for(int i=0; i<2; ++i) {
            glSetFrameOfReferenceF(T_gw[i].inverse());
            glDrawAxis(10);
            glUnsetFrameOfReference();
        }

//        glViewport(DESIRED_WIDTH,0,DESIRED_WIDTH,DESIRED_HEIGHT);

        // clear glViewport
        glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);
    }

    GLWindow window;
    CameraDevice camera;

    std::vector<rpg::ImageWrapper> img;
    GLuint m_glTex[2];
    int width, height;
    std::mutex imageMutex;

    Sophus::SE3 T_gw[2];

    bool shouldQuit;
};

int main (int argc, char** argv){
    Application app;
    return app.Run();
}

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

const int DESIRED_WIDTH = 694;
const int DESIRED_HEIGHT = 518;

const int WINDOW_WIDTH  = 2 * DESIRED_WIDTH;
const int WINDOW_HEIGHT = 2 * DESIRED_HEIGHT;

inline void glDrawTexturesQuad(float t, float b, float l, float r)
{
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2f(l,b);
    glTexCoord2f(1,0); glVertex2f(r,b);
    glTexCoord2f(1,1); glVertex2f(r,t);
    glTexCoord2f(0,1); glVertex2f(l,t);
    glEnd();
    glDisable(GL_TEXTURE_2D);
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
            camera.SetProperty("ImageBinningX", 2);
            camera.SetProperty("ImageBinningY", 2);
            camera.SetProperty("ImageWidth", 694);
            camera.SetProperty("ImageHeight", 518);
            if(!camera.InitDriver( "AlliedVision" )) {
                cerr << "Couldn't start driver for camera " << endl;
                exit(1);
            }
            camera.Capture(img);
        }
        width = img[0].width();
        height = img[0].height();



        // Setup grid
        window.AddChildToRoot(new GLGrid());

        // Setup OpenGL Textures
        // Create two OpenGL textures for stereo images
        glGenTextures(2, m_glTex);

        // Allocate texture memory on GPU
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        glPixelStorei(GL_PACK_ALIGNMENT,1);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);

        // Setup Tracker objects
        // Unit hell!
        const double ppi = 72; // Points Per Inch
        const double USwp = 11 * ppi;
        const double UShp = 8.5 * ppi;
        const double mpi = 0.0254; // meters per inch
        const double mpp = mpi / ppi; // meters per point
        const double unit = mpp; //1; //mpp;

        // Setup Trackers
        for(int i=0; i<2; ++i) {
            tracker[i] = new Tracker(width,height);
            tracker[i]->target.GenerateRandom(60,unit*USwp*25/(842.0),unit*USwp*75/(842.0),unit*USwp*40/(842.0),Eigen::Vector2d(unit*USwp,unit*UShp));
//            tracker[i]->target.SaveEPS("stereo.eps");
        }

        Matrix<double,9,1> camParamsVec; // = Var<Matrix<double,9,1> >("cam_params");
        camParamsVec << 0.808936, 1.06675, 0.495884, 0.520504, 0, 0, 0, 0, 0.0;
        camParams = MatlabCamera( width,height, width*camParamsVec[0],height*camParamsVec[1], width*camParamsVec[2],height*camParamsVec[3], camParamsVec[4], camParamsVec[5], camParamsVec[6], camParamsVec[7], camParamsVec[8]);

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
        while(!shouldQuit) {
//            imageMutex.lock();
            camera.Capture(img);
//            imageMutex.unlock();

            for(int i=0; i<2; ++i) {
                tracker[i]->ProcessFrame(camParams,img[i].Image.data);
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
        glClearColor (0.0, 0.0, 0.0, 0.0);
//        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_LIGHTING);
        glDisable(GL_COLOR_MATERIAL );

        // Draw 3D stuff using the ModelView Matrix from SimpleGUI
        glViewport(0,0,DESIRED_WIDTH,DESIRED_HEIGHT);
        DrawTarget(tracker[0]->target,Eigen::Vector2d(0,0),1,0.2,0.2);
        for(int i=0; i<2; ++i) {
            glSetFrameOfReferenceF(tracker[i]->T_hw.inverse());
            glDrawAxis(0.2);
            glUnsetFrameOfReference();
        }

        // Reset model view matrices for displaying images
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        // Draw Threshold images for cam 0
        glViewport(DESIRED_WIDTH,0,DESIRED_WIDTH,DESIRED_HEIGHT);
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_LUMINANCE,GL_UNSIGNED_BYTE,tracker[0]->tI.get());
        glColor3f (1.0, 1.0, 1.0);
        glDrawTexturesQuad(-1,1,-1,1);

        // Upload textures for images
        if( img.size() >= 2 ) {
            imageMutex.lock();
            glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_LUMINANCE,GL_UNSIGNED_BYTE,img[0].Image.data);
            glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_LUMINANCE,GL_UNSIGNED_BYTE,img[1].Image.data);
            imageMutex.unlock();
        }

        glOrtho(-0.5,width-0.5,height-0.5,-0.5,0,1.0);

        // Draw left image
        glViewport(0,DESIRED_HEIGHT,DESIRED_WIDTH,DESIRED_HEIGHT);
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glColor3f (1.0, 1.0, 1.0);
        glDrawTexturesQuad(height,0,0,width);
        for( int i=0; i<tracker[0]->conics.size(); ++i ) {
          glColorBin(tracker[0]->conics_target_map[i],tracker[0]->target.circles3D().size());
          DrawCross(tracker[0]->conics[i].center,2);
        }
//        glDrawTexturesQuad(-1,1,-1,1);

        // Draw right image
        glViewport(DESIRED_WIDTH,DESIRED_HEIGHT,DESIRED_WIDTH,DESIRED_HEIGHT);
        glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
        glColor3f (1.0, 1.0, 1.0);
        glDrawTexturesQuad(height,0,0,width);
        for( int i=0; i<tracker[1]->conics.size(); ++i ) {
          glColorBin(tracker[1]->conics_target_map[i],tracker[1]->target.circles3D().size());
          DrawCross(tracker[1]->conics[i].center,2);
        }

        glDisable(GL_TEXTURE_2D);

        // Reset OpenGL to what SimpleGui expects
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
//        glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);
        glViewport(0,0,DESIRED_WIDTH,DESIRED_HEIGHT);

    }

    GLWindow window;
    CameraDevice camera;

    std::vector<rpg::ImageWrapper> img;
    GLuint m_glTex[2];
    int width, height;
    std::mutex imageMutex;

    MatlabCamera camParams;
    Tracker* tracker[2];

    bool shouldQuit;
};

int main (int argc, char** argv){
    Application app;
    return app.Run();
}

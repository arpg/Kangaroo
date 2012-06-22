#include <thread>
#include <future>
#include <mutex>

#include <CVars/CVar.h>

#include <SimpleGui/Gui.h>
#include <SimpleGui/GetPot>
#include <SimpleGui/GLMesh.h>

#include <RPG/Devices/Camera/CameraDevice.h>
#include <fiducials/camera.h>

#include <sophus/se3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


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

    void InitCamera()
    {
        // Setup Camera device
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
        w = img[0].width();
        h = img[0].height();

        // Setup camera parameters
        VectorXd camParamsVec(6); // = Var<Matrix<double,9,1> >("cam_params");
        camParamsVec << 0.558526, 0.747774, 0.484397, 0.494393, -0.249261, 0.0825967;
        camParams = MatlabCamera( w,h, camParamsVec);

        // Setup stereo baseline
        Eigen::Matrix3d R_rl;
        R_rl << 0.999995,   0.00188482,  -0.00251896,
                -0.0018812,     0.999997,   0.00144025,
                0.00252166,  -0.00143551,     0.999996;

        Eigen::Vector3d l_r;
        l_r <<    -0.203528, -0.000750334, 0.00403201;

        T_rl = Sophus::SE3(R_rl, l_r);
    }

    void InitOpenGLTextures()
    {
        // Create two OpenGL textures for stereo images
        glGenTextures(2, m_glTex);

        // Allocate texture memory on GPU
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        glPixelStorei(GL_PACK_ALIGNMENT,1);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    }

    void InitCudaAndCudaMem()
    {
        // Use first Graphics device for Cuda with GL Interop
        cudaGLSetGLDevice(0);

        cudaMalloc(&d_lookup, w*h*sizeof(float2));
        for(int i=0; i<2; ++i ) {
            cudaMalloc(&d_raw[i], w*h*sizeof(uchar1));
            cudaMalloc(&d_rect[i], w*h*sizeof(uchar1));
        }

        // make lookup table based on camera parameters
    }

    void Init()
    {
        InitCamera();
        InitOpenGLTextures();

        // Setup floor grid
        window.AddChildToRoot(new GLGrid());

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
            camera.Capture(img);
            for(int i=0; i<2; ++i) {
                cudaMemcpy(d_raw[i],img[i].Image.data,w*h*sizeof(uchar1),cudaMemcpyHostToDevice);
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

        // Reset model view matrices for displaying images
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        // Draw Threshold images for cam 0
        glViewport(DESIRED_WIDTH,0,DESIRED_WIDTH,DESIRED_HEIGHT);

        // Upload textures for images
        if( img.size() >= 2 ) {
            glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_LUMINANCE,GL_UNSIGNED_BYTE,img[0].Image.data);
            glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_LUMINANCE,GL_UNSIGNED_BYTE,img[1].Image.data);
        }

        glOrtho(-0.5,w-0.5,h-0.5,-0.5,0,1.0);

        // Draw left image
        glViewport(0,DESIRED_HEIGHT,DESIRED_WIDTH,DESIRED_HEIGHT);
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glColor3f (1.0, 1.0, 1.0);
        glDrawTexturesQuad(h,0,0,w);

        // Draw right image
        glViewport(DESIRED_WIDTH,DESIRED_HEIGHT,DESIRED_WIDTH,DESIRED_HEIGHT);
        glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
        glColor3f (1.0, 1.0, 1.0);
        glDrawTexturesQuad(h,0,0,w);

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
    int w, h;

    MatlabCamera camParams;
    Sophus::SE3 T_rl;

    bool shouldQuit;

    float2* d_lookup;
    uchar1* d_raw[2];
    uchar1* d_rect[2];

};

int main (int argc, char** argv){
    Application app;
    return app.Run();
}

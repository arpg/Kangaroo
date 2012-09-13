#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>

#include <CVars/CVar.h>

#include <RPG/Devices/Camera/CameraDevice.h>
#include "common/RpgCameraOpen.h"

#include <fiducials/tracker.h>
#include <fiducials/drawing.h>

#include <pangolin/pangolin.h>

#include "common/StereoIntrinsicsOptimisation.h"
#include "common/DisplayUtils.h"

using namespace std;
using namespace Eigen;
using namespace pangolin;

const int UI_WIDTH = 150;

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

//CVarUtils::CreateCVar<ConsoleFunc>("TestConsoleFunc",&TestConsoleFunc);
//bool TestConsoleFunc( std::vector<std::string> *args)
//{
//    for(int i=0; i<args->size(); ++i ) {
//        cout << args->at(i) << endl;
//    }
//}

class Application
{
public:
    Application()
        : // window(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, __FILE__ ),
          shouldQuit(false)
    {
    }

    ~Application()
    {
    }

    void InitCamera(int argc, char** argv)
    {
        OpenRpgCamera(argc,argv);

        camera.Capture(img);
        width = img[0].width();
        height = img[0].height();

        // Setup camera parameters
        VectorXd camParamsVec(6); // = Var<Matrix<double,9,1> >("cam_params");
        camParamsVec << 0.558526, 0.747774, 0.484397, 0.494393, -0.249261, 0.0825967;
        camParams = MatlabCamera( width,height, camParamsVec);

        // Setup stereo baseline
        Eigen::Matrix3d R_rl;
        R_rl << 0.999995,   0.00188482,  -0.00251896,
                -0.0018812,     0.999997,   0.00144025,
                0.00252166,  -0.00143551,     0.999996;

        Eigen::Vector3d l_r;
        l_r <<    -0.203528, -0.000750334, 0.00403201;

        T_rl = Sophus::SE3(R_rl, l_r);
    }

    void InitTrackers()
    {
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
        }
        tracker[0]->target.SaveRotatedEPS("stereo.eps", 1.0/unit);
    }

    void OptimiseRun()
    {
        Eigen::Matrix<double,3,Eigen::Dynamic> pattern = tracker[0]->TargetPattern3D();

        while(!shouldQuit) {
            if(keyframes.size() > 2 ) {
                OptimiseIntrinsicsPoses(pattern,keyframes,camParams, T_rl);
            }else{
                usleep(1000);
            }
        }
    }

    int Run(int argc, char** argv)
    {
        InitCamera(argc,argv);
        InitTrackers();

        // Create Graphics Context using Glut
        pangolin::CreateGlutWindowAndBind(__FILE__,width*2+UI_WIDTH, height*2);
        glewInit();

        // Setup default OpenGL parameters
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        glEnable(GL_LINE_SMOOTH);
        glPixelStorei(GL_PACK_ALIGNMENT,1);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);

        // Tell the base view to arrange its children equally
        pangolin::CreatePanel("ui")
            .SetBounds(0.0, 1.0, 0.0, Attach::Pix(UI_WIDTH));

        View& container = CreateDisplay()
                .SetBounds(0,1.0, Attach::Pix(UI_WIDTH), 1.0)
                .SetLayout(LayoutEqual);

        const int N = 4;
        for(int i=0; i<N; ++i ) {
            View& disp = CreateDisplay().SetAspect((double)width/height);
            container.AddDisplay(disp);
        }

        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam;
        s_cam.Set(ProjectionMatrix(width,height,420,420,width/2,height/2,0.1,1000));
        s_cam.Set(IdentityMatrix(GlModelViewStack));
        container[2].SetHandler(new Handler3D(s_cam));

        GlTexture tex8(width,height,GL_LUMINANCE8);

        boost::thread trackerThreads[2];

        // Run Optimisation Loop
        boost::thread optThread( boost::bind( &Application::OptimiseRun, this ) );

        Var<bool> run("ui.Run", true, true);
        Var<bool> step("ui.Step", false, false);
        Var<bool> save_kf("ui.Save Keyframe", false, false);

        // Run main loop
        for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
        {
            const bool go = frame==0 || run || Pushed(step);

            if(go) {
                // Capture camera images
                camera.Capture(img);
            }

            // Track
            for(int i=0; i<2; ++i) {
//                tracker[i]->ProcessFrame(camParams,img[i].Image.data);
                trackerThreads[i] = boost::thread(boost::bind(&Tracker::ProcessFrame, tracker[i], camParams,img[i].Image.data) );
            }
            for(int i=0; i<2; ++i) {
                trackerThreads[i].join();
            }

            if( Pushed(save_kf) ) {
                cout << "Save Keyframe" << endl;

                if(keyframes.size() == 0) {
                    // First keyframe. Initialise baseline estimate
                    T_rl = tracker[1]->T_hw * tracker[0]->T_hw.inverse();
                }

                StereoKeyframe kf;
                for(int i=0; i<2; ++i) {
                    kf.obs[i] = tracker[i]->TargetPatternObservations();
                    kf.T_fw[i] = tracker[i]->T_hw;
                }
                kfMutex.lock();
                keyframes.push_back(kf);
                kfMutex.unlock();
            }

            // Display
            pangolin::DisplayBase().ActivateScissorAndClear();

            // Draw Stereo images
            for(int c=0; c<2; ++c ) {
                container[c].Activate();
                tex8.Upload(img[c].Image.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
                glColor3f(1,1,1);
                tex8.RenderToViewportFlipY();

                glOrtho(-0.5,width-0.5,height-0.5,-0.5,0,1.0);
                for( int i=0; i<tracker[c]->conics.size(); ++i ) {
                    glBinColor(tracker[c]->conics_target_map[i],tracker[c]->target.circles3D().size());
                    DrawCross(tracker[c]->conics[i].center,2);
                }
            }

            // Draw Threshold image
            container[3].Activate();
            glColor3f(1,1,1);
            tex8.Upload(tracker[0]->tI.get(), GL_LUMINANCE, GL_UNSIGNED_BYTE);
            tex8.RenderToViewportFlipY();

            // Draw current tracker poses
            container[2].ActivateAndScissor(s_cam);
            glEnable(GL_DEPTH_TEST);

            DrawTarget(tracker[0]->target,Eigen::Vector2d(0,0),1,0.2,0.2);
            for(int i=0; i<2; ++i) {
                glSetFrameOfReferenceF(tracker[i]->T_hw.inverse());
                glDrawAxis(0.2);
                glUnsetFrameOfReference();
            }

            // Draw Stereo keyframes
            for(size_t kf=0; kf < keyframes.size(); ++kf ) {
                glSetFrameOfReferenceF(keyframes[kf].T_fw[0].inverse());
                glDrawAxis(0.05);
                glUnsetFrameOfReference();

                glSetFrameOfReferenceF(keyframes[kf].T_fw[1].inverse());
                glDrawAxis(0.05);
                glUnsetFrameOfReference();
            }

            pangolin::RenderViews();
            pangolin::FinishGlutFrame();
        }

//        window.Run();
        shouldQuit = true;

        optThread.join();
    }

    CameraDevice camera;

    std::vector<rpg::ImageWrapper> img;
    GLuint m_glTex[2];
    int width, height;
    boost::mutex kfMutex;

    MatlabCamera camParams;
    Tracker* tracker[2];

    Sophus::SE3 T_rl;
    std::vector<StereoKeyframe> keyframes;

    bool shouldQuit;
};

int main (int argc, char** argv){
    Application app;
    return app.Run(argc,argv);
}

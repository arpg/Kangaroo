#include <pangolin/pangolin.h>
#include "common/ViconTracker.h"

// TODO: Work out why this is required to build in Debug
#undef GL_VERSION_2_0
#undef GL_VERSION_2_1
#undef GL_VERSION_3_0
#undef GL_ARB_gpu_shader_fp64
#include <unsupported/Eigen/OpenGLSupport>

using namespace std;
using namespace pangolin;
using namespace Eigen;

int main( int /*argc*/, char* argv[] )
{
    const int w = 640;
    const int h = 480;

    pangolin::CreateGlutWindowAndBind(__FILE__,w,h);

    glEnable (GL_LINE_SMOOTH);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam;
    s_cam.SetProjectionMatrix(ProjectionMatrix(640,480,420,420,320,240,0.1,1000));
    s_cam.SetModelViewMatrix(ModelViewLookAt(0,5,5,0,0,0,0,0,1));

    View& d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
      .SetHandler(new Handler3D(s_cam,AxisZ));

    ViconTracking tracker("CAR","192.168.10.1");

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        d_cam.ActivateScissorAndClear(s_cam);

        glColor3f(0.8,0.8,0.8);
        glDraw_z0(1.0,5);
        glDisable(GL_DEPTH_TEST);
        glDrawAxis(2);
        glEnable(GL_DEPTH_TEST);

        // Draw Vicon
        glPushMatrix();
        glMultMatrix( tracker.T_wf.matrix() );
        glDrawAxis(1);
        glPopMatrix();

        pangolin::FinishGlutFrame();
    }
}

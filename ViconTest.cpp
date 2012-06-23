#include <pangolin/pangolin.h>
#include "ViconTracker.h"

// TODO: Work out why this is required to build in Debug
#undef GL_VERSION_2_0
#undef GL_VERSION_2_1
#undef GL_VERSION_3_0
#undef GL_ARB_gpu_shader_fp64
#include <unsupported/Eigen/OpenGLSupport>

using namespace std;
using namespace pangolin;
using namespace Eigen;

inline void glSetFrameOfReferenceF( const Sophus::SE3& T_wf )
{
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrix( T_wf.matrix() );
}

inline void glUnsetFrameOfReference()
{
  glPopMatrix();
}

inline void glDrawGrid(float num_lines, float line_delta)
{
    glBegin(GL_LINES);

    for(int i = -num_lines; i < num_lines; i++){
        glVertex3f( line_delta*num_lines, i*line_delta, 0.0);
        glVertex3f(-line_delta*num_lines, i*line_delta, 0.0);

        glVertex3f(i*line_delta,  line_delta*num_lines, 0.0);
        glVertex3f(i*line_delta, -line_delta*num_lines, 0.0);
    }

    glEnd();
}

inline void glDrawAxis(float s)
{
  glBegin(GL_LINES);
  glColor3f(1,0,0);
  glVertex3f(0,0,0);
  glVertex3f(s,0,0);
  glColor3f(0,1,0);
  glVertex3f(0,0,0);
  glVertex3f(0,s,0);
  glColor3f(0,0,1);
  glVertex3f(0,0,0);
  glVertex3f(0,0,s);
  glEnd();
}

int main( int /*argc*/, char* argv[] )
{
    const int w = 640;
    const int h = 480;

    pangolin::CreateGlutWindowAndBind("Main",w,h);

    glEnable (GL_LINE_SMOOTH);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam;
    s_cam.Set(ProjectionMatrix(640,480,420,420,320,240,0.1,1000));
    s_cam.Set(IdentityMatrix(GlModelViewStack));

    View& d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
      .SetHandler(new Handler3D(s_cam));

    ViconTracking tracker("CAR","192.168.10.1");

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        d_cam.ActivateScissorAndClear(s_cam);

        glColor3f(0.5,0.5,0.5);
        glDrawGrid(20,0.25);

        glDisable(GL_DEPTH_TEST);
        glColor3f(0.8,0.8,0.8);
        glDrawGrid(5,1.0);
        glDrawAxis(2);
        glEnable(GL_DEPTH_TEST);

        // Draw Vicon
        glSetFrameOfReferenceF(tracker.T_wf);
        glDrawAxis(1);
        glUnsetFrameOfReference();


        pangolin::FinishGlutFrame();
    }
}

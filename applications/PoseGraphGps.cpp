#include <pangolin/pangolin.h>
#include <sophus/se3.h>
#include <SceneGraph/SceneGraph.h>

#include "common/PoseGraph.h"
#include "common/GLPoseGraph.h"
#include "common/LoadPosesFromFile.h"
#include "common/GLCameraHistory.h"

using namespace std;
using namespace pangolin;
using namespace Eigen;

int main( int /*argc*/, char* argv[] )
{
    const int w = 640;
    const int h = 480;

    pangolin::CreateGlutWindowAndBind(__FILE__,w,h);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

    // Define 3D Objects
    SceneGraph::GLSceneGraph glGraph;
    SceneGraph::GLGrid glGrid(50,10.0, true);
    glGrid.SetPosition(0,0,40);
    glGraph.AddChild(&glGrid);

    // RDF transforms
    Eigen::Matrix3d RDFvision;RDFvision<< 1,0,0,  0,1,0,  0,0,1;
    Eigen::Matrix3d RDFrobot; RDFrobot << 0,1,0,  0,0, 1,  1,0,0;
    Eigen::Matrix4d T_vis_ro = Eigen::Matrix4d::Identity();
    T_vis_ro.block<3,3>(0,0) = RDFvision.transpose() * RDFrobot;
    Eigen::Matrix4d T_ro_vis = Eigen::Matrix4d::Identity();
    T_ro_vis.block<3,3>(0,0) = RDFrobot.transpose() * RDFvision;

    // Load Visual Odometry
    SceneGraph::GLCameraHistory hist_vis_odometry;
    hist_vis_odometry.LoadFromTimeAbsoluteCartesianFile("/Users/slovegrove/data/Monument/Trajectory_ts_2laps.txt" );
//    glGraph.AddChild(&hist_vis_odometry);

    // Load GPS
    SceneGraph::GLCameraHistory hist_gps;
    hist_gps.LoadFromTimeLatLon("/Users/slovegrove/data/Monument/gps_tab.txt" );
    glGraph.AddChild(&hist_gps);

    cout << "Visual Odometry Edges " << hist_vis_odometry.m_T_on.size() << endl;
    cout << "GPS Edges " << hist_gps.m_T_on.size() << endl;

    // Define pose graph problem
    PoseGraph posegraph;

    GLPoseGraph glposegraph(posegraph);
    glGraph.AddChild(&glposegraph);

    int gpsid = 0;
    double gpstime = hist_gps.m_time_s[gpsid];

    while(gpstime < hist_vis_odometry.m_time_s[0]) {
        gpstime = hist_gps.m_time_s[++gpsid];
    }

    // Populate Pose Graph
    for( int i=0; /*i < 1000 &&*/ i < hist_vis_odometry.m_T_on.size(); ++i )
    {
        const double vistime = hist_vis_odometry.m_time_s[i];

        int kfid = -1;

        if(i == 0 ) {
            Keyframe* kf = new Keyframe();
            kfid = posegraph.AddKeyframe(kf);
        }else{
            kfid = posegraph.AddRelativeKeyframe(i-1, Sophus::SE3d(hist_vis_odometry.m_T_on[i]) );
        }

        if(gpstime < vistime) {
            Eigen::Vector3d xyz_m = hist_gps.m_T_wh[gpsid].col(3).head<3>();
            posegraph.AddUnaryEdge(kfid, xyz_m );
            while(gpstime < vistime) {
                gpstime = hist_gps.m_time_s[++gpsid];
            }
        }
    }

    // Define OpenGL Render state
    pangolin::OpenGlRenderState stacks3d;
    stacks3d.SetProjectionMatrix(ProjectionMatrix(640,480,420,420,320,240,0.1,1E6));
    stacks3d.SetModelViewMatrix(ModelViewLookAt(0,5,-5,0,0,0,0,0,-1));

    // Create Interactive view of data
    View& d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
      .SetHandler(new Handler3D(stacks3d,AxisNegZ))
      .SetDrawFunction(SceneGraph::ActivateDrawFunctor(glGraph, stacks3d));

    pangolin::RegisterKeyPressCallback(' ', [&posegraph]() {posegraph.Start();} );

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        d_cam.ActivateScissorAndClear(stacks3d);

        pangolin::FinishGlutFrame();
    }

    posegraph.Stop();
    exit(0);
}

#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
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
    glDisable( GL_LIGHTING );

    // Define 3D Objects
    SceneGraph::GLSceneGraph glGraph;
    SceneGraph::GLAxis glAxis;
    SceneGraph::GLGrid glGrid(5,1, true);
//    glGrid.SetPosition(0,0,40);
    glGraph.AddChild(&glGrid);
    glGraph.AddChild(&glAxis);

    // RDF transforms
    Eigen::Matrix3d RDFvision;RDFvision<< 1,0,0,  0,1,0,  0,0,1;
    Eigen::Matrix3d RDFrobot; RDFrobot << 0,1,0,  0,0, 1,  1,0,0;
    Eigen::Matrix3d RDFvicon; RDFvicon << 0,-1,0,  0,0, -1,  1,0,0;
    Eigen::Matrix4d T_vis_ro = Eigen::Matrix4d::Identity();
    T_vis_ro.block<3,3>(0,0) = RDFvision.transpose() * RDFrobot;
    Eigen::Matrix4d T_ro_vis = Eigen::Matrix4d::Identity();
    T_ro_vis.block<3,3>(0,0) = RDFrobot.transpose() * RDFvision;
    Eigen::Matrix4d T_vis_vic = Eigen::Matrix4d::Identity();
    T_vis_vic.block<3,3>(0,0) = RDFvision.transpose() * RDFvicon;
    Eigen::Matrix4d T_vic_vis = Eigen::Matrix4d::Identity();
    T_vic_vis.block<3,3>(0,0) = RDFvicon.transpose() * RDFvision;

    // Load Visual Odometry
    SceneGraph::GLCameraHistory hist_vis_odometry;
//    hist_vis_odometry.LoadFromAbsoluteCartesianFile("/Users/slovegrove/data/Monument/Trajectory_ts_2laps.txt", 0, 1E6, Matrix4d::Identity(), Matrix4d::Identity() );
    hist_vis_odometry.LoadFromRelativeCartesianFile("/Users/slovegrove/code/kangaroo/build/applications/mocap_icp.txt",0, 1E6/*, T_ro_vis, T_vis_ro*/ );
    glGraph.AddChild(&hist_vis_odometry);

    SceneGraph::GLCameraHistory hist_vicon;
    hist_vicon.LoadFromAbsoluteCartesianFile("/Users/slovegrove/code/Dev/Loggers/ImageLogger/build/kinect_vicon/Vicon.txt" );
    glGraph.AddChild(&hist_vicon);

    // Define pose graph problem
    PoseGraph posegraph;

    GLPoseGraph glposegraph(posegraph);
    glGraph.AddChild(&glposegraph);

//    const Sophus::SE3d T_hz(Sophus::SO3d(0.2,0.1,0.1), Eigen::Vector3d(0.3,0.2,0.1) );
//    int coord_z = posegraph.AddSecondaryCoordinateFrame(T_vis_ro);
    int coord_z = posegraph.AddSecondaryCoordinateFrame( Sophus::SE3d(T_ro_vis) );

    const int num_poses = std::min((unsigned long)1000, std::min(hist_vis_odometry.m_T_on.size(), hist_vicon.m_T_wh.size()));
    hist_vis_odometry.SetNumberToShow(num_poses);
    hist_vicon.SetNumberToShow(num_poses);
    cout << "Poses: " << num_poses << endl;

//    // Populate Pose Graph
//    for( int i=0; /*i < 1000 &&*/ i < num_poses; ++i )
//    {
//        int kfid = -1;

//        if(i == 0 ) {
//            Keyframe* kf = new Keyframe();
//            kfid = posegraph.AddKeyframe(kf);
//        }else{
//            kfid = posegraph.AddRelativeKeyframe(i-1, Sophus::SE3d(hist_vis_odometry.m_T_on[i]) );
//        }

////        posegraph.AddIndirectUnaryEdge(kfid, coord_z, Sophus::SE3d(hist_vis_odometry.m_T_wh[i] ) * T_hz );
//        posegraph.AddIndirectUnaryEdge(kfid, coord_z, Sophus::SE3d(hist_vicon.m_T_wh[i] )  );
//    }

    // Populate Pose Graph
    for( int i=0; /*i < 1000 &&*/ i < num_poses; ++i )
    {
        Keyframe* kf = new Keyframe(Sophus::SE3d(hist_vicon.m_T_wh[i]) );
        const int kfid = posegraph.AddKeyframe(kf);
        posegraph.AddUnaryEdge(kfid, Sophus::SE3d(hist_vicon.m_T_wh[i] )  );
        if(i > 0 ) {
            posegraph.AddIndirectBinaryEdge(kfid-1, kfid, coord_z, Sophus::SE3d(hist_vis_odometry.m_T_on[i]) );
        }
    }


    pangolin::RegisterKeyPressCallback(' ', [&posegraph]() {posegraph.Start();} );
    pangolin::RegisterKeyPressCallback('v', [&hist_vicon]() {hist_vicon.SetVisible(!hist_vicon.IsVisible());});

    // Define OpenGL Render state
    pangolin::OpenGlRenderState stacks3d;
    stacks3d.SetProjectionMatrix(ProjectionMatrix(640,480,420,420,320,240,0.1,1E6));
    stacks3d.SetModelViewMatrix(ModelViewLookAt(0,5,5,0,0,0,0,0,1));


    // Create Interactive view of data
    View& d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
      .SetHandler(new Handler3D(stacks3d))
      .SetDrawFunction(SceneGraph::ActivateDrawFunctor(glGraph, stacks3d));

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        d_cam.ActivateScissorAndClear(stacks3d);

        pangolin::FinishGlutFrame();
    }

    cout << posegraph.GetSecondaryCoordinateFrame(coord_z).GetT_wk().matrix3x4() << endl;
    posegraph.Stop();

    exit(0);
}

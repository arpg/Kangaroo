#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>

#include <pangolin/pangolin.h>
#include <Sophus/se3.h>
#include <SceneGraph/SceneGraph.h>

#include "common/CeresQuatXYZW.h"
#include "common/LoadPosesFromFile.h"
#include "common/GLCameraHistory.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <boost/ptr_container/ptr_vector.hpp>

using namespace std;
using namespace pangolin;
using namespace Eigen;
using namespace ceres;

double* pt(Sophus::SE3& T) {
    return T.translation().data();
}

double* pq(Sophus::SE3& T) {
    return T.so3().unit_quaternion().coeffs().data();
}

const double* pt(const Sophus::SE3& T) {
    return T.translation().data();
}

const double* pq(const Sophus::SE3& T) {
    return T.so3().unit_quaternion().coeffs().data();
}

// translation, quaternion (x,y,z,  w, i, j, k)
struct Keyframe {
    Keyframe() {
    }

    Keyframe(Sophus::SE3 T_wk)
        : m_T_wk(T_wk)
    {
    }

    Sophus::SE3 GetT_wk() const {
        return m_T_wk;
    }

    void SetT_wk(Sophus::SE3 T_wk) {
        m_T_wk = T_wk;
    }

    Sophus::SE3 m_T_wk;
};

struct BinaryEdgeXYZQuatCostFunction
//    :   ceres::SizedCostFunction<7,7>
{
    BinaryEdgeXYZQuatCostFunction( Sophus::SE3 T_ba)
        : m_T_ba(T_ba)
    {
    }

    template <typename T>
    bool operator()(const T* const R_wb, const T* const t_wb, const T* const R_wa, const T* const t_wa, T* residuals) const
    {
        const double* _R_ba = pq(m_T_ba);
        const double* _t_ba = pt(m_T_ba);
        const T R_measured_ba[4] = {(T)_R_ba[0],(T)_R_ba[1],(T)_R_ba[2],(T)_R_ba[3] };
        const T t_measured_ba[3] = {(T)_t_ba[0],(T)_t_ba[1],(T)_t_ba[2] };

        T R_ab[4]; T t_ab[3];
        XYZUnitQuatXYZWInverseCompose(R_wa, t_wa, R_wb, t_wb, R_ab, t_ab);

        T R_a_ma[4]; T t_a_ma[3];
        XYZUnitQuatXYZWCompose(R_ab, t_ab, R_measured_ba, t_measured_ba, R_a_ma, t_a_ma);

        residuals[0] = t_a_ma[0] / (T)10;
        residuals[1] = t_a_ma[1] / (T)10;
        residuals[2] = t_a_ma[2] / (T)10;
        QuatXYZWToAngleAxis(R_a_ma, residuals+3);
        return true;
    }

    // Observed transformation
    Sophus::SE3 m_T_ba;
};

class PoseGraph {
public:
    PoseGraph() {
        quat_param = new QuatXYZWParameterization;
    }

    int AddKeyframe(Keyframe* kf)
    {
        int id = keyframes.size();
        keyframes.push_back( kf );
        problem.AddParameterBlock(pq(kf->m_T_wk), 4, quat_param);
        problem.AddParameterBlock(pt(kf->m_T_wk), 3, NULL);
        return id;
    }

    int AddKeyframe()
    {
        AddKeyframe(new Keyframe() );
    }

    Keyframe& GetKeyframe(int a) {
        assert(a < keyframes.size());
        return keyframes[a];
    }

    void AddBinaryEdge(int b, int a, Sophus::SE3 T_ba)
    {
        Keyframe& kfa = GetKeyframe(a);
        Keyframe& kfb = GetKeyframe(b);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<BinaryEdgeXYZQuatCostFunction, 6, 4, 3, 4, 3>(
                new BinaryEdgeXYZQuatCostFunction(T_ba)
            ), NULL,
            pq(kfb.m_T_wk), pt(kfb.m_T_wk),
            pq(kfa.m_T_wk), pt(kfa.m_T_wk)
        );
    }

    int AddRelativeKeyframe(int keyframe_a, Sophus::SE3 T_ak)
    {
        const int k = AddKeyframe();
        Keyframe& kf_a = GetKeyframe(keyframe_a);
        Keyframe& kf_k = GetKeyframe(k);

        // Initialise keyframes pose based on relative transform
//        kf_k.SetT_wk( kf_a.GetT_wk() * T_ak );
        kf_k.SetT_wk(kf_a.GetT_wk() * Sophus::SE3(Sophus::SO3(), Vector3d(0.1,0,0)));

        AddBinaryEdge(keyframe_a,k,T_ak);
        return k;
    }

    void Solve()
    {
        ceres::Solver::Options options;
//        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 8;
//        options.check_gradients = true;
        options.update_state_every_iteration = true;
        options.max_num_iterations = 1000;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;
    }

//protected:
    LocalParameterization* quat_param;
    boost::ptr_vector<Keyframe> keyframes;
    ceres::Problem problem;
};

class GLPoseGraph : public SceneGraph::GLObject
{
public:
    GLPoseGraph(const PoseGraph& posegraph)
        : posegraph(posegraph)
    {
    }

    void DrawCanonicalObject()
    {
        // Draw each keyframe
        const int N = posegraph.keyframes.size();
        for(int i=0; i < N; ++i ) {
            const Keyframe& kf = posegraph.keyframes[i];
            glPushMatrix();
            glMultMatrix(kf.GetT_wk().matrix());
            SceneGraph::GLAxis::DrawUnitAxis();
            glPopMatrix();
        }
    }

    const PoseGraph& posegraph;
};


int main( int /*argc*/, char* argv[] )
{
    const int w = 640;
    const int h = 480;

    pangolin::CreateGlutWindowAndBind(__FILE__,w,h);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

    // Define 3D Objects
    SceneGraph::GLSceneGraph glGraph;
    SceneGraph::GLGrid glGrid(50,2.0, true);
    SceneGraph::GLAxis glAxis;
    glGraph.AddChild(&glGrid);
    glGraph.AddChild(&glAxis);

    // RDF transforms
    Eigen::Matrix3d RDFvision;RDFvision<< 1,0,0,  0,1,0,  0,0,1;
    Eigen::Matrix3d RDFrobot; RDFrobot << 0,1,0,  0,0, 1,  1,0,0;
    Eigen::Matrix4d T_vis_ro = Eigen::Matrix4d::Identity();
    T_vis_ro.block<3,3>(0,0) = RDFvision.transpose() * RDFrobot;
    Eigen::Matrix4d T_ro_vis = Eigen::Matrix4d::Identity();
    T_ro_vis.block<3,3>(0,0) = RDFrobot.transpose() * RDFvision;

    // Load Visual Odometry
    SceneGraph::GLCameraHistory hist_vis_odometry;
    hist_vis_odometry.LoadFromTimeAbsoluteCartesianFile("/Users/slovegrove/data/Monument/Trajectory_ts_1lap.txt", 0, 1E6, Matrix4d::Identity(), Matrix4d::Identity() );
    glGraph.AddChild(&hist_vis_odometry);

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

        if(i == 0 ) {
            Keyframe* kf = new Keyframe();
            posegraph.AddKeyframe(kf);
            posegraph.problem.SetParameterBlockConstant(pq(kf->m_T_wk));
            posegraph.problem.SetParameterBlockConstant(pt(kf->m_T_wk));
        }else{
            posegraph.AddRelativeKeyframe(i-1, Sophus::SE3(hist_vis_odometry.m_T_on[i]) );
        }

        if(gpstime < vistime) {
//            posegraph.AddUnaryEdge;
            while(gpstime < vistime) {
                gpstime = hist_gps.m_time_s[++gpsid];
            }
        }
    }

    boost::thread optThread( boost::bind( &PoseGraph::Solve, &posegraph ) );
//    posegraph.Solve();

    // Define OpenGL Render state
    pangolin::OpenGlRenderState stacks3d;
    stacks3d.SetProjectionMatrix(ProjectionMatrix(640,480,420,420,320,240,0.1,1E6));
    stacks3d.SetModelViewMatrix(ModelViewLookAt(0,5,5,0,0,0,0,0,1));

    // Create Interactive view of data
    View& d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
      .SetHandler(new Handler3D(stacks3d,AxisZ))
      .SetDrawFunction(SceneGraph::ActivateDrawFunctor(glGraph, stacks3d));

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        d_cam.ActivateScissorAndClear(stacks3d);

        pangolin::FinishGlutFrame();
    }

    optThread.interrupt();
    exit(0);
}

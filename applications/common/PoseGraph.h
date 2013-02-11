#pragma once

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <boost/ptr_container/ptr_vector.hpp>

#include <sophus/se3.h>
#include "CeresQuatXYZW.h"

double* pt(Sophus::SE3d& T) {
    return T.translation().data();
}

double* pq(Sophus::SE3d& T) {
    return const_cast<double*>(T.so3().unit_quaternion().coeffs().data());
}

const double* pt(const Sophus::SE3d& T) {
    return const_cast<double*>(T.translation().data());
}

const double* pq(const Sophus::SE3d& T) {
    return T.so3().unit_quaternion().coeffs().data();
}

// translation, quaternion (x,y,z,  w, i, j, k)
struct Keyframe {
    Keyframe() {
    }

    Keyframe(Sophus::SE3d T_wk)
        : m_T_wk(T_wk)
    {
    }

    Sophus::SE3d& GetT_wk() {
        return m_T_wk;
    }

    Sophus::SE3d GetT_wk() const {
        return m_T_wk;
    }

    void SetT_wk(Sophus::SE3d T_wk) {
        m_T_wk = T_wk;
    }

    Sophus::SE3d m_T_wk;
};

struct UnaryEdge6DofCostFunction
{
    UnaryEdge6DofCostFunction( Sophus::SE3d T_wk )
        : m_T_wk(T_wk)
    {
    }

    template <typename T>
    bool operator()(const T* const R_wk, const T* const t_wk, T* residuals) const
    {
        const double* _R_wk = pq(m_T_wk);
        const double* _t_wk = pt(m_T_wk);
        const T measR_wk[4] = {(T)_R_wk[0],(T)_R_wk[1],(T)_R_wk[2],(T)_R_wk[3] };
        const T meast_wk[3] = {(T)_t_wk[0],(T)_t_wk[1],(T)_t_wk[2] };
        XYZUnitQuatXYZWPoseResidual(R_wk, t_wk, measR_wk, meast_wk, residuals);
        return true;
    }

    Sophus::SE3d m_T_wk;
};

// Indirect measurement of T_wk through T_wz given frame transform T_zk
struct UnaryEdgeIndirect6DofCostFunction
{
    UnaryEdgeIndirect6DofCostFunction( Sophus::SE3d T_wz )
        : m_T_wz(T_wz)
    {
    }

    template <typename T>
    bool operator()(const T* const R_kz, const T* const t_kz, const T* const R_wk, const T* const t_wk, T* residuals) const
    {
        const double* _R_wz = pq(m_T_wz);
        const double* _t_wz = pt(m_T_wz);
        const T measR_wz[4] = {(T)_R_wz[0],(T)_R_wz[1],(T)_R_wz[2],(T)_R_wz[3] };
        const T meast_wz[3] = {(T)_t_wz[0],(T)_t_wz[1],(T)_t_wz[2] };

        T measR_wk[4];
        T meast_wk[3];

        XYZUnitQuatXYZWComposeInverse( measR_wz, meast_wz, R_kz, t_kz, measR_wk, meast_wk);
        XYZUnitQuatXYZWPoseResidual(R_wk, t_wk, measR_wk, meast_wk, residuals);
        return true;
    }

    Sophus::SE3d m_T_wz;
};

struct UnaryEdgeXYCostFunction
{
    UnaryEdgeXYCostFunction( Eigen::Vector3d xy )
        : mk_w(xy)
    {
    }

    template <typename T>
    bool operator()(const T* const t_wa, T* residuals) const
    {
        residuals[0] = (T)1E-2 * ((T)mk_w(0) - t_wa[0]);
        residuals[1] = (T)1E-2 * ((T)mk_w(1) - t_wa[1]);
        residuals[2] = (T)1E-2 * ((T)mk_w(2) - t_wa[2]);
        return true;
    }

    Eigen::Vector3d mk_w;
};

struct BinaryEdgeXYZQuatCostFunction
{
    BinaryEdgeXYZQuatCostFunction( Sophus::SE3d T_ba)
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

        T R_ba[4]; T t_ba[3];
        XYZUnitQuatXYZWInverseCompose(R_wb, t_wb, R_wa, t_wa, R_ba, t_ba);
        XYZUnitQuatXYZWPoseResidual(R_ba, t_ba, R_measured_ba, t_measured_ba, residuals);
        return true;
    }

    // Observed transformation
    Sophus::SE3d m_T_ba;
};

struct BinaryEdgeXYZQuatIndirectCostFunction
{
    BinaryEdgeXYZQuatIndirectCostFunction( Sophus::SE3d T_zb_za)
        : m_T_zb_za(T_zb_za)
    {
    }

    template <typename T>
    bool operator()(const T* const R_kz, const T* const t_kz, const T* const R_wb, const T* const t_wb, const T* const R_wa, const T* const t_wa, T* residuals) const
    {
        const double* _R_zb_za = pq(m_T_zb_za);
        const double* _t_zb_za = pt(m_T_zb_za);
        const T measR_zb_za[4] = {(T)_R_zb_za[0],(T)_R_zb_za[1],(T)_R_zb_za[2],(T)_R_zb_za[3] };
        const T meast_zb_za[3] = {(T)_t_zb_za[0],(T)_t_zb_za[1],(T)_t_zb_za[2] };


        T measR_ba[4]; T meast_ba[3];
        XYZUnitQuatXYZWChangeFrame(measR_zb_za,meast_zb_za, R_kz, t_kz, measR_ba, meast_ba);

        T R_ba[4]; T t_ba[3];
        XYZUnitQuatXYZWInverseCompose(R_wb, t_wb, R_wa, t_wa, R_ba, t_ba);

        XYZUnitQuatXYZWPoseResidual(R_ba, t_ba, measR_ba, meast_ba, residuals);
        return true;
    }

    // Observed transformation
    Sophus::SE3d m_T_zb_za;
};

class PoseGraph {
public:
    PoseGraph()
        : problem(0), running(false)
    {
        Clear();
    }

    ~PoseGraph()
    {
        delete problem;
    }

    void Clear()
    {
        if(problem) {
            delete problem;
        }
        problem = new ceres::Problem();
        quat_param = new QuatXYZWParameterization;
        huber_loss = new ceres::HuberLoss(0.01);
        cauchy_loss = new ceres::CauchyLoss(0.001);

        keyframes.clear();
        coord_frames.clear();
    }


    int AddKeyframe(Keyframe* kf)
    {
        const int id = keyframes.size();
        keyframes.push_back( kf );
        problem->AddParameterBlock(pq(kf->m_T_wk), 4, quat_param);
        problem->AddParameterBlock(pt(kf->m_T_wk), 3, NULL);
        return id;
    }

    int AddKeyframe()
    {
        return AddKeyframe(new Keyframe() );
    }

    Keyframe& GetKeyframe(int a)
    {
        assert(a < keyframes.size());
        return keyframes[a];
    }

    int AddSecondaryCoordinateFrame(Sophus::SE3d T_kz = Sophus::SE3() )
    {
        Keyframe* kf = new Keyframe(T_kz);
        const int id = coord_frames.size();
        coord_frames.push_back( kf );
        problem->AddParameterBlock(pq(kf->m_T_wk), 4, quat_param);
        problem->AddParameterBlock(pt(kf->m_T_wk), 3, NULL);
        return id;
    }

    Keyframe& GetSecondaryCoordinateFrame(int z)
    {
        assert(z < coord_frames.size());
        return coord_frames[z];
    }

    void AddBinaryEdge(int b, int a, Sophus::SE3d T_ba)
    {
        Keyframe& kfa = GetKeyframe(a);
        Keyframe& kfb = GetKeyframe(b);

        problem->AddResidualBlock(
            new ceres::AutoDiffCostFunction<BinaryEdgeXYZQuatCostFunction, 6, 4, 3, 4, 3>(
                new BinaryEdgeXYZQuatCostFunction(T_ba)
            ), cauchy_loss,
            pq(kfb.m_T_wk), pt(kfb.m_T_wk),
            pq(kfa.m_T_wk), pt(kfa.m_T_wk)
        );
    }

    void AddUnaryEdge(int a, Eigen::Vector3d xyz)
    {
        Keyframe& kfa = GetKeyframe(a);

        problem->AddResidualBlock(
            new ceres::AutoDiffCostFunction<UnaryEdgeXYCostFunction, 3, 3>(
                new UnaryEdgeXYCostFunction(xyz)
            ), NULL,
            pt(kfa.m_T_wk)
        );
    }

    void AddUnaryEdge(int a, Sophus::SE3d T_wa)
    {
        Keyframe& kfa = GetKeyframe(a);

        problem->AddResidualBlock(
            new ceres::AutoDiffCostFunction<UnaryEdge6DofCostFunction, 6, 4, 3>(
                new UnaryEdge6DofCostFunction(T_wa)
            ), NULL,
            pq(kfa.m_T_wk),
            pt(kfa.m_T_wk)
        );
    }

    int AddRelativeKeyframe(int keyframe_a, Sophus::SE3d T_ak)
    {
        const int k = AddKeyframe();
        Keyframe& kf_a = GetKeyframe(keyframe_a);
        Keyframe& kf_k = GetKeyframe(k);

        // Initialise keyframes pose based on relative transform
        kf_k.SetT_wk( kf_a.GetT_wk() * T_ak );
        AddBinaryEdge(keyframe_a,k,T_ak);
        return k;
    }

    void AddIndirectUnaryEdge(int kf_a, int coord_z, Sophus::SE3d T_w_az)
    {
        Keyframe& coz = GetSecondaryCoordinateFrame(coord_z);
        Keyframe& kfa = GetKeyframe(kf_a);

        problem->AddResidualBlock(
            new ceres::AutoDiffCostFunction<UnaryEdgeIndirect6DofCostFunction, 6, 4, 3, 4, 3>(
                new UnaryEdgeIndirect6DofCostFunction(T_w_az)
            ), cauchy_loss,
            pq(coz.m_T_wk), pt(coz.m_T_wk),
            pq(kfa.m_T_wk), pt(kfa.m_T_wk)
        );

//        // Only optimise rotation
//        problem->SetParameterBlockConstant( pt(coz.m_T_wk) );
//        problem->SetParameterBlockConstant( pq(coz.m_T_wk) );
    }

    void AddIndirectBinaryEdge(int b, int a, int coord_z, Sophus::SE3d T_bz_az)
    {
        Keyframe& coz = GetSecondaryCoordinateFrame(coord_z);
        Keyframe& kfa = GetKeyframe(a);
        Keyframe& kfb = GetKeyframe(b);

        problem->AddResidualBlock(
            new ceres::AutoDiffCostFunction<BinaryEdgeXYZQuatIndirectCostFunction, 6, 4, 3, 4, 3, 4, 3>(
                new BinaryEdgeXYZQuatIndirectCostFunction(T_bz_az)
            ), NULL,
            pq(coz.m_T_wk), pt(coz.m_T_wk),
            pq(kfb.m_T_wk), pt(kfb.m_T_wk),
            pq(kfa.m_T_wk), pt(kfa.m_T_wk)
        );
    }


    void SetSecondaryCoordinateFrameFreedom(int coord_z, bool rot_free, bool trans_free) {
        Keyframe& coz = GetSecondaryCoordinateFrame(coord_z);

        if(rot_free) {
            problem->SetParameterBlockVariable( pq(coz.m_T_wk) );
        }else{
            problem->SetParameterBlockConstant( pq(coz.m_T_wk) );
        }
        if(trans_free) {
            problem->SetParameterBlockVariable( pt(coz.m_T_wk) );
        }else{
            problem->SetParameterBlockConstant( pt(coz.m_T_wk) );
        }
    }

    void SetKeyframeFreedom(int kf_id, bool rot_free, bool trans_free) {
        Keyframe& kf = GetKeyframe(kf_id);

        if(rot_free) {
            problem->SetParameterBlockVariable( pq(kf.m_T_wk) );
        }else{
            problem->SetParameterBlockConstant( pq(kf.m_T_wk) );
        }
        if(trans_free) {
            problem->SetParameterBlockVariable( pt(kf.m_T_wk) );
        }else{
            problem->SetParameterBlockConstant( pt(kf.m_T_wk) );
        }
    }

    void Solve()
    {
        ceres::Solver::Options options;
//        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 4;
//        options.check_gradients = true;
        options.update_state_every_iteration = true;
        options.max_num_iterations = 1000;
//        options.gradient_tolerance = 1E-50;
        options.function_tolerance = 1E-50;
        options.parameter_tolerance = 1E-50;

        ceres::Solver::Summary summary;
        ceres::Solve(options, problem, &summary);
        std::cout << summary.FullReport() << std::endl;
        running = false;
    }

    void Start() {
        if(!running) {
            running = true;
            optThread = boost::thread(boost::bind( &PoseGraph::Solve, this )) ;
        }
    }

    void Stop() {
        optThread.interrupt();
    }

//protected:
    ceres::HuberLoss* huber_loss;
    ceres::CauchyLoss* cauchy_loss;
    ceres::LocalParameterization* quat_param;
    boost::ptr_vector<Keyframe> keyframes;
    boost::ptr_vector<Keyframe> coord_frames;
    ceres::Problem* problem;
    boost::thread optThread;
    bool running;
};

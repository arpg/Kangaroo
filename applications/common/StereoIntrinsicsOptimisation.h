#pragma once

#include <Eigen/Eigen>

#include "CamParam.h"
#include "LeastSquaresBlockHelpers.h"

struct StereoKeyframe
{
    Sophus::SE3 T_fw[2];
    Eigen::Matrix<double,2,Eigen::Dynamic> obs[2];
};

static inline Eigen::Matrix<double,2,3> dpi_dx(const Eigen::Vector3d& x)
{
    const double x2x2 = x(2)*x(2);
    Eigen::Matrix<double,2,3> ret;
    ret << 1.0 / x(2), 0,  -x(0) / x2x2,
            0, 1.0 / x(2), -x(1) / x2x2;
    return ret;
}

static inline Eigen::Matrix<double,4,4> se3_gen(unsigned i) {

    Eigen::Matrix<double,4,4> ret;
    ret.setZero();

    switch(i) {
    case 0: ret(0,3) = 1; break;
    case 1: ret(1,3) = 1; break;
    case 2: ret(2,3) = 1; break;
    case 3: ret(1,2) = -1; ret(2,1) = 1; break;
    case 4: ret(0,2) = 1; ret(2,0) = -1; break;
    case 5: ret(0,1) = -1; ret(1,0) = 1; break;
    }

    return ret;
}

static void OptimiseIntrinsicsPoses(
    Eigen::Matrix<double,3,Eigen::Dynamic> pattern,
    std::vector<StereoKeyframe>& keyframes,
    MatlabCamera& cam,
    Sophus::SE3& T_rl
) {
    // keyframes.size might increase asynchronously, so save
    const int N = keyframes.size();

    typedef CamParamMatlab<MatlabCamera> CamParam;
    const int PARAMS_K = CamParam::PARAMS;
    const int PARAMS_T = 6;
    const int PARAMS_TOTAL = PARAMS_K + (1+N)* PARAMS_T;

    unsigned int num_obs = 0;
    double sumsqerr = 0;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> JTJ(PARAMS_TOTAL,PARAMS_TOTAL);
    Eigen::Matrix<double,Eigen::Dynamic,1> JTy(PARAMS_TOTAL);
    JTJ.setZero();
    JTy.setZero();

    // Make JTJ and JTy from observations over each Keyframe
    for( size_t kf=0; kf < N; ++kf ) {
        // For each observation
        for( size_t on=0; on < pattern.cols(); ++on ) {
            // Construct block contributions for JTJ and JTy
            const Sophus::SE3 T_lt = keyframes[kf].T_fw[0];

            const Eigen::Vector3d Pt = pattern.col(on);
            const Eigen::Vector3d Pl = T_lt * Pt;

            const Eigen::Vector2d obsl = keyframes[kf].obs[0].col(on);
            if( std::isfinite(obsl[0]) ) {
                const Eigen::Vector2d pl_ = project(Pl);
                const Eigen::Vector2d pl = cam.map(pl_);
                const Eigen::Vector2d errl = pl - obsl;
                sumsqerr += errl.squaredNorm();
                num_obs++;

                Eigen::Matrix<double,2,PARAMS_K> Jk = CamParam::dmap_by_dk(cam,pl_);

                const Eigen::Matrix<double,2,3> dpi = dpi_dx(Pl);
                const Eigen::Matrix<double,2,2> dmap = CamParam::dmap_by_duv(cam,pl_);
                const Eigen::Matrix<double,2,3> dmapdpi = dmap * dpi;
                const Eigen::Matrix<double,2,4> dmapdpiTlt = dmapdpi * T_lt.matrix3x4();

                Eigen::Matrix<double,2,PARAMS_T> J_T_lw;
                for(int i=0; i<PARAMS_T; ++i ) {
                    J_T_lw.col(i) = dmapdpiTlt * se3_gen(i) * unproject(Pt);
                }

                AddSparseOuterProduct<double,2,PARAMS_K,PARAMS_T>(
                    JTJ,JTy,  Jk,0,  J_T_lw,PARAMS_K+(1+kf)*PARAMS_T, errl
                );
            }

            const Eigen::Vector2d obsr = keyframes[kf].obs[1].col(on);
            if(std::isfinite(obsr[0])) {
                const Eigen::Vector3d Pr = T_rl * Pl;
                const Eigen::Vector2d pr_ = project(Pr);
                const Eigen::Vector2d pr  = cam.map(pr_);
                const Eigen::Vector2d errr = pr - obsr;
                sumsqerr += errr.squaredNorm();
                num_obs++;

                Eigen::Matrix<double,2,PARAMS_K> Jk = CamParam::dmap_by_dk(cam,pr_);

                const Eigen::Matrix<double,2,3> dpi = dpi_dx(Pr);
                const Eigen::Matrix<double,2,2> dmap = CamParam::dmap_by_duv(cam,pr_);
                const Eigen::Matrix<double,2,3> dmapdpi = dmap * dpi;
                const Eigen::Matrix<double,2,4> dmapdpiT_rl = dmapdpi * T_rl.matrix3x4();

                Eigen::Matrix<double,2,PARAMS_T> J_T_rl;
                Eigen::Matrix<double,2,PARAMS_T> J_T_lw;
                for(int i=0; i<PARAMS_T; ++i ) {
                    J_T_rl.col(i) = dmapdpiT_rl * se3_gen(i) * T_lt.matrix() * unproject(Pt);
                    J_T_lw.col(i) = dmapdpiT_rl * T_lt.matrix() * se3_gen(i) * unproject(Pt);
                }

                AddSparseOuterProduct<double,2,PARAMS_K,PARAMS_T,PARAMS_T>(
                    JTJ,JTy,  Jk,0,  J_T_rl,PARAMS_K,  J_T_lw,PARAMS_K+(1+kf)*PARAMS_T, errr
                );
            }
        }
    }

    std::cout << "=============== RMSE: " << sqrt(sumsqerr/num_obs) << " ====================" << std::endl;

    Eigen::FullPivLU<Eigen::MatrixXd> lu_JTJ(JTJ);
    Eigen::Matrix<double,Eigen::Dynamic,1> x = -1.0 * lu_JTJ.solve(JTy);

    if( x.norm() > 1 ) {
        x = x / x.norm();
    }

    if( lu_JTJ.rank() == PARAMS_TOTAL )
    {
        CamParam::UpdateCam(cam, x.head<PARAMS_K>());

        // Update baseline
        T_rl = T_rl * Sophus::SE3::exp(x.segment<PARAMS_T>(PARAMS_K) );

        // Update poses
        for( size_t kf=0; kf < N; ++kf ) {
            keyframes[kf].T_fw[0] = keyframes[kf].T_fw[0] *
                Sophus::SE3::exp(x.segment<PARAMS_T>(PARAMS_K + (1+kf)*PARAMS_T));
            keyframes[kf].T_fw[1] = T_rl * keyframes[kf].T_fw[0];
        }

        std::cout << cam << std::endl;
        std::cout << T_rl.matrix() << std::endl;
    }else{
        std::cerr << "Rank deficient! Missing: " << (PARAMS_TOTAL - lu_JTJ.rank()) << std::endl;
        std::cerr << lu_JTJ.kernel() << std::endl;
    }
}

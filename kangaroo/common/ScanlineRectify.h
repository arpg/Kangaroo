#pragma once

#include <Eigen/Eigen>

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

#include "CameraModelPyramid.h"
#include "BaselineFromCamModel.h"

#include <kangaroo/kangaroo.h>

inline Eigen::Matrix3d MakeK(const Eigen::VectorXd& camParamsVec, size_t w, size_t h)
{
    Eigen::Matrix3d K;
    K << camParamsVec(0)*w, 0, camParamsVec(2)*w,
            0, camParamsVec(1)*h, camParamsVec(3)*h,
            0,0,1;
    return K;
}

inline Sophus::SE3d CreateScanlineRectifiedLookupAndT_rl(
    roo::Image<float2> dlookup_left, roo::Image<float2> dlookup_right,
    const Sophus::SE3d T_rl,
    const Eigen::Matrix3d& lK, double lk1, double lk2,
    const Eigen::Matrix3d& rK, double rk1, double rk2
) {
    Eigen::Matrix3d lKinv = MakeKinv(lK);
    Eigen::Matrix3d rKinv = MakeKinv(rK);

    const Sophus::SO3d R_rl = T_rl.so3();
    const Sophus::SO3d R_lr = R_rl.inverse();
    const Eigen::Vector3d l_r = T_rl.translation();
    const Eigen::Vector3d r_l = - (R_lr * l_r);

    // Current up vector for each camera (in left FoR)
    const Eigen::Vector3d lup_l = Eigen::Vector3d(0,1,0);
    const Eigen::Vector3d rup_l = R_lr * Eigen::Vector3d(0,1,0);

    // Hypothetical fwd vector for each camera, perpendicular to baseline (in left FoR)
    const Eigen::Vector3d lfwd = lup_l.cross(r_l);
    const Eigen::Vector3d rfwd = rup_l.cross(r_l);

    // New fwd is average of left / right hypothetical baselines (also perpendicular to baseline)
    const Eigen::Vector3d new_fwd = (lfwd + rfwd).normalized();

    // Define new basis (in left FoR);
    const Eigen::Vector3d x = r_l.normalized();
    const Eigen::Vector3d z = -new_fwd;
    const Eigen::Vector3d y  = z.cross(x).normalized();

    // New orientation for both left and right cameras (expressed relative to original left)
    Eigen::Matrix3d mR_nl;
    mR_nl << x, y, z;

    // By definition, the right camera now lies exactly on the x-axis with the same orientation
    // as the left camera.
    const Sophus::SE3d T_nr_nl = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d(-r_l.norm(),0,0) );

    // Homographies which should be applied to left and right images to scan-line rectify them
    const Eigen::Matrix3d Hl_nl = lK * mR_nl.transpose() * lKinv;
    const Eigen::Matrix3d Hr_nr = rK * (mR_nl * R_lr.matrix()).transpose() * rKinv;

    // Copy to simple Array objects to pass to CUDA by Value
    roo::Mat<float,9> H_ol_nl;
    roo::Mat<float,9> H_or_nr;

    for(int r=0; r<3; ++r) {
        for(int c=0; c<3; ++c) {
            H_ol_nl[3*r+c] = Hl_nl(r,c);
            H_or_nr[3*r+c] = Hr_nr(r,c);
        }
    }

    // Invoke CUDA Kernel to generate lookup table
    roo::CreateMatlabLookupTable(dlookup_left, lK(0,0), lK(1,1), lK(0,2), lK(1,2), lk1, lk2, H_ol_nl);
    roo::CreateMatlabLookupTable(dlookup_right,rK(0,0), rK(1,1), rK(0,2), rK(1,2), rk1, rk2, H_or_nr);

    return T_nr_nl;
}

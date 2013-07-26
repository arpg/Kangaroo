#pragma once

#include <calibu/Calibu.h>

inline Sophus::SE3d T_rlFromCamModelRDF(const calibu::CameraModelAndTransform& lcmod, const calibu::CameraModelAndTransform& rcmod, const Eigen::Matrix3d& targetRDF)
{
    // Transformation matrix to adjust to target RDF
    Eigen::Matrix4d Tadj[2] = {Eigen::Matrix4d::Identity(),Eigen::Matrix4d::Identity()};
    Tadj[0].block<3,3>(0,0) = targetRDF.transpose() * lcmod.camera.RDF();
    Tadj[1].block<3,3>(0,0) = targetRDF.transpose() * rcmod.camera.RDF();

    // Computer Poses in our adjust coordinate system
    const Eigen::Matrix4d T_lw_ = Tadj[0] * lcmod.T_wc.matrix().inverse();
    const Eigen::Matrix4d T_rw_ = Tadj[1] * rcmod.T_wc.matrix().inverse();

    // Computer transformation to right camera frame from left
    const Eigen::Matrix4d T_rl = T_rw_ * T_lw_.inverse();

    return Sophus::SE3d(T_rl.block<3,3>(0,0), T_rl.block<3,1>(0,3) );
}

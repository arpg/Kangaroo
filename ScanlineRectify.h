#ifndef SCANLINERECTIFY_H
#define SCANLINERECTIFY_H

#include <Eigen/Eigen>

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

#include "CudaImage.h"
#include "kernel.h"

inline void CamModelScaleToDimensions(mvl::CameraModel& camModel, int w, int h)
{
    const double scale = w / camModel.Width();

    if(scale != 1.0) {
        mvl_camera_t* cam = camModel.GetModel();

        cam->linear.width  = w;
        cam->linear.height = h;

        cam->linear.fx *= scale;
        cam->linear.fy *= scale;
        cam->linear.cx = scale*(cam->linear.cx+0.5) - 0.5;
        cam->linear.cy = scale*(cam->linear.cy+0.5) - 0.5;

        if(camModel.Type() == MVL_CAMERA_WARPED ) {
            // MV_CAMERA_WARPED specific params still apply
        }else if(camModel.Type() == MVL_CAMERA_LUT ) {
            std::cerr << "Can't Modify LUT to match image size" << std::endl;
            // Can't update the camera params easily
            exit(1);
        }
    }
}

inline void CamModelCropToRegionOfInterest(mvl::CameraModel& camModel, const NppiRect& roi)
{
    mvl_camera_t* cam = camModel.GetModel();
    cam->linear.cx -= roi.x;
    cam->linear.cy -= roi.y;
}

inline Eigen::Matrix3d ScaleK(const Eigen::Matrix3d& K, double imageScale)
{
    Eigen::Matrix3d rK = K;
    rK(0,0) *= imageScale;
    rK(1,1) *= imageScale;
    rK(0,2) = imageScale * (K(0,2)+0.5) - 0.5;
    rK(1,2) = imageScale * (K(1,2)+0.5) - 0.5;
    return rK;
}

inline Eigen::Matrix3d MakeK(const Eigen::VectorXd& camParamsVec, size_t w, size_t h)
{
    Eigen::Matrix3d K;
    K << camParamsVec(0)*w, 0, camParamsVec(2)*w,
            0, camParamsVec(1)*h, camParamsVec(3)*h,
            0,0,1;
    return K;
}

inline Eigen::Matrix3d MakeKinv(const Eigen::Matrix3d& K)
{
    Eigen::Matrix3d Kinv = Eigen::Matrix3d::Identity();
    Kinv << 1.0/K(0,0), 0, - K(0,2) / K(0,0),
            0, 1.0/K(1,1), - K(1,2) / K(1,1),
            0,0,1;
    return Kinv;
}

Sophus::SE3 CreateScanlineRectifiedLookupAndT_rl(
    Gpu::Image<float2> dlookup_left, Gpu::Image<float2> dlookup_right,
    const Sophus::SE3 T_rl, const Eigen::Matrix3d& K, double kappa1, double kappa2,
    size_t w, size_t h
) {
//    Eigen::Matrix3d K =    MakeK(camParamsVec, w, h);
    Eigen::Matrix3d Kinv = MakeKinv(K);

    const Sophus::SO3 R_rl = T_rl.so3();
    const Sophus::SO3 R_lr = R_rl.inverse();
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
    const Sophus::SE3 T_nr_nl = Sophus::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(-r_l.norm(),0,0) );


    // Homographies which should be applied to left and right images to scan-line rectify them
    const Eigen::Matrix3d Hl_nl = K * mR_nl.transpose() * Kinv;
    const Eigen::Matrix3d Hr_nr = K * (mR_nl * R_lr.matrix()).transpose() * Kinv;

    // Copy to simple Array objects to pass to CUDA by Value
    Gpu::Array<float,9> H_ol_nl;
    Gpu::Array<float,9> H_or_nr;

    for(int r=0; r<3; ++r) {
        for(int c=0; c<3; ++c) {
            H_ol_nl[3*r+c] = Hl_nl(r,c);
            H_or_nr[3*r+c] = Hr_nr(r,c);
        }
    }

    // Invoke CUDA Kernel to generate lookup table
    CreateMatlabLookupTable(dlookup_left, K(0,0), K(1,1), K(0,2), K(1,2), kappa1, kappa2, H_ol_nl);
    CreateMatlabLookupTable(dlookup_right,K(0,0), K(1,1), K(0,2), K(1,2), kappa1, kappa2, H_or_nr);

    return T_nr_nl;
}

Sophus::SE3 T_rlFromCamModelRDF(const mvl::CameraModel& lcmod, const mvl::CameraModel& rcmod, const Eigen::Matrix3d& targetRDF)
{
    // Transformation matrix to adjust to target RDF
    Eigen::Matrix4d Tadj[2] = {Eigen::Matrix4d::Identity(),Eigen::Matrix4d::Identity()};
    Tadj[0].block<3,3>(0,0) = targetRDF.transpose() * lcmod.RDF();
    Tadj[1].block<3,3>(0,0) = targetRDF.transpose() * rcmod.RDF();

    // Computer Poses in our adjust coordinate system
    const Eigen::Matrix4d T_lw_ = Tadj[0] * lcmod.GetPose().inverse();
    const Eigen::Matrix4d T_rw_ = Tadj[1] * rcmod.GetPose().inverse();

    // Computer transformation to right camera frame from left
    const Eigen::Matrix4d T_rl = T_rw_ * T_lw_.inverse();

    return Sophus::SE3(T_rl.block<3,3>(0,0), T_rl.block<3,1>(0,3) );
}


#endif // SCANLINERECTIFY_H

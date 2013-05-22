#pragma once

#include <Sophus/se3.hpp>
#include <kangaroo/Sdf.h>

#include "SaveGIL.h"
#include "../MarchingCubes.h"

struct KinectKeyframe
{
    KinectKeyframe(int w, int h, Sophus::SE3d T_iw)
        : img(w,h), T_iw(T_iw)
    {
    }

    Sophus::SE3d T_iw;
    Gpu::Image<uchar3, Gpu::TargetDevice, Gpu::Manage> img;
};

void SaveMeshlab(Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage>& vol, boost::ptr_vector<KinectKeyframe>& keyframes, float fu, float fv, float u0, float v0)
{
    Eigen::Matrix3d RDFvision;  RDFvision  << 1,0,0,  0,1,0,  0,0,1;
    Eigen::Matrix3d RDFmeshlab; RDFmeshlab << 1,0,0,  0,-1,0, 0,0,-1;
    Eigen::Matrix4d T_vis_ml = Eigen::Matrix4d::Identity();
    T_vis_ml.block<3,3>(0,0) = RDFvision.transpose() * RDFmeshlab;
    Eigen::Matrix4d T_ml_vis = Eigen::Matrix4d::Identity();
    T_ml_vis.block<3,3>(0,0) = RDFmeshlab.transpose() * RDFvision;

    const float f = fu / 10.0;
    const float pu = f / fu;
    const float pv = f / fv;

    std::string mesh_filename = "mesh";
    std::ofstream of("project.mlp");

    of << "<!DOCTYPE MeshLabDocument>" << std::endl;
    of << "<MeshLabProject>" << std::endl;


    of << " <MeshGroup>" << std::endl;

    Gpu::SaveMesh(mesh_filename, vol);
    of << "  <MLMesh label=\"mesh.ply\" filename=\"" << mesh_filename << ".ply\">" << std::endl;
    of << "   <MLMatrix44>" << std::endl;
    of << "1 0 0 0 " << std::endl;
    of << "0 1 0 0 " << std::endl;
    of << "0 0 1 0 " << std::endl;
    of << "0 0 0 1 " << std::endl;
    of << "</MLMatrix44>" << std::endl;
    of << "  </MLMesh>" << std::endl;

    of << " </MeshGroup>" << std::endl;

    of << " <RasterGroup>" << std::endl;
    for(int i=0; i<keyframes.size(); ++i) {
        const KinectKeyframe& kf = keyframes[i];
        const Eigen::Matrix4d T = T_ml_vis * kf.T_iw.matrix();
//        const Eigen::Matrix4d T = kf.T_iw.inverse().matrix() * T_vis_gl;
//        const Eigen::Matrix3d R = T.block<3,3>(0,0);
//        const Eigen::Vector3d t = T.block<3,1>(0,3);
        std::ostringstream oss; oss << "keyframe_" << i << ".png";
        std::string img_filename = oss.str();
        SaveGIL(img_filename,kf.img );
        of << "  <MLRaster label=\"" << img_filename << "\">" << std::endl;
//        of << "   <VCGCamera TranslationVector=\"" << t.transpose() << " 1\" LensDistortion=\"0 0\" ViewportPx=\"" << kf.img.w << " " << kf.img.h << "\" PixelSizeMm=\"" << pu << " " << pv << "\" CenterPx=\"" << u0 << " " << v0 << "\" FocalMm=\"" << f << "\" RotationMatrix=\"" << R(0,0) << " " << R(0,1) << " " << R(0,2) << " 0 " << R(1,0) << " " << R(1,1) << " " << R(1,2) << " 0 " << R(2,0) << " " << R(2,1) << " " << R(2,2) << " 0 0 0 0 1 \"/>" << std::endl;
        of << "   <VCGCamera TranslationVector=\"0 0 0 1\" LensDistortion=\"0 0\" ViewportPx=\"" << kf.img.w << " " << kf.img.h << "\" PixelSizeMm=\"" << pu << " " << pv << "\" CenterPx=\"" << u0 << " " << v0 << "\" FocalMm=\"" << f << "\" RotationMatrix=\"" << T(0,0) << " " << T(0,1) << " " << T(0,2) << " " << T(0,3) << " " << T(1,0) << " " << T(1,1) << " " << T(1,2) << " " << T(1,3) << " " << T(2,0) << " " << T(2,1) << " " << T(2,2) << " " << T(2,3) << " 0 0 0 1 \"/>" << std::endl;
        of << "   <Plane semantic=\"\" fileName=\"" << img_filename << "\"/>" << std::endl;
        of << "  </MLRaster>" << std::endl;
    }
    of << " </RasterGroup>" << std::endl;

    of << "</MeshLabProject> " << std::endl;

    of.close();
}

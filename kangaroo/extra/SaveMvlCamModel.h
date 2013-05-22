#pragma once

#include <calibu/cam/CameraModel.h>
#include <Eigen/Eigen>
#include <mvl++/Mvlpp/SE3.h>

inline void SaveCamModelRobotConvention(std::string filename, std::string name, int w, int h, double fu, double fv, double u0, double v0, double sx, double k1, double k2, double p1, double p2, double k3, Eigen::Matrix4d T_wc)
{
    std::ofstream f;
    f.open(filename);

    if(f.is_open()) {
        f.precision(10);
        f.setf(std::ios::fixed, std::ios::floatfield);
        const Eigen::Matrix<double,6,1> x = mvl::T2Cart(T_wc);
        f << "<camera_model name=\"" << name << "\" index=\"0\" serialno=\"-1\" type=\"MVL_CAMERA_WARPED\" version=\"7\">" << std::endl;
        f << "<pose>"<< x(0) << ";"<< x(1) << ";"<< x(2) << ";"<< x(3) << ";"<< x(4) << ";"<< x(5) << "</pose>" << std::endl;
        f << "<right> 0; 1; 0 </right>" << std::endl;
        f << "<down> 0; 0; 1 </down>" << std::endl;
        f << "<forward> 1; 0; 0</forward>" << std::endl;
        f << "<width>"<< w << "</width>" << std::endl;
        f << "<height>"<< h << "</height>" << std::endl;
        f << "<fx>"<< fu << "</fx>" << std::endl;
        f << "<cx>"<< u0 << "</cx>" << std::endl;
        f << "<fy>"<< fv << "</fy>" << std::endl;
        f << "<cy>"<< v0 << "</cy>" << std::endl;
        f << "<sx>"<< sx << "</sx>" << std::endl;
        f << "<kappa1>"<< k1 << "</kappa1>" << std::endl;
        f << "<kappa2>"<< k2 << "</kappa2>" << std::endl;
        f << "<kappa3>"<< k3 << "</kappa3>" << std::endl;
        f << "<tau1>"<< p1 << "</tau1>" << std::endl;
        f << "<tau2>"<< p2 << "</tau2>" << std::endl;
        f << "</camera_model>" << std::endl;
        f.close();
    }else{
        std::cerr << "Unable to save camera file " << filename << std::endl;
    }
}

inline void SaveCamModelLeftRightVisionConvention(
    std::string filename_prefix, int w, int h,
    double lfu, double lfv, double lu0, double lv0, double lsx, double lk1, double lk2, double lp1, double lp2, double lk3,
    double rfu, double rfv, double ru0, double rv0, double rsx, double rk1, double rk2, double rp1, double rp2, double rk3,
    Eigen::Matrix4d T_lr
) {
    Eigen::Matrix3d RDFvision;RDFvision<< 1,0,0,  0,1,0,  0,0,1;
    Eigen::Matrix3d RDFrobot; RDFrobot << 0,1,0,  0,0, 1,  1,0,0;
    Eigen::Matrix4d T_vis_ro = Eigen::Matrix4d::Identity();
    T_vis_ro.block<3,3>(0,0) = RDFvision.transpose() * RDFrobot;
    Eigen::Matrix4d T_ro_vis = Eigen::Matrix4d::Identity();
    T_ro_vis.block<3,3>(0,0) = RDFrobot.transpose() * RDFvision;

    Eigen::Matrix4d Trobot_lr = T_ro_vis*T_lr*T_vis_ro;
    SaveCamModelRobotConvention(filename_prefix + "lcmod.xml", "left",  w,h,lfu,lfv,lu0,lv0,lsx,lk1,lk2,lp1,lp2,lk3, Eigen::Matrix4d::Identity() );
    SaveCamModelRobotConvention(filename_prefix + "rcmod.xml", "right", w,h,rfu,rfv,ru0,rv0,rsx,rk1,rk2,rp1,rp2,rk3, Trobot_lr );
}

inline void SaveCamModelLeftRightVisionConvention(std::string filename_prefix, const calibu::CameraModelT<calibu::Poly>& cam, Eigen::Matrix4d T_lr)
{
    const double sx = 0;
    SaveCamModelLeftRightVisionConvention(
        filename_prefix, cam.Width(), cam.Height(),
        cam.data()[0], cam.data()[1], cam.data()[2], cam.data()[3], 0,  cam.data()[4], cam.data()[5], 0, 0, cam.data()[6],
        cam.data()[0], cam.data()[1], cam.data()[2], cam.data()[3], 0,  cam.data()[4], cam.data()[5], 0, 0, cam.data()[6],
        T_lr
    );
}


inline void SaveCamModelRobotConvention(std::string filename, std::string name, MatlabCamera cam, Eigen::Matrix4d T_wc)
{
    std::ofstream f;
    f.open(filename);

    if(f.is_open()) {
        const Eigen::Vector6d x = mvl::T2Cart(T_wc);
        f << "<camera_model name=\"" << name << "\" index=\"0\" serialno=\"-1\" type=\"MVL_CAMERA_WARPED\" version=\"7\">" << std::endl;
        f << "<pose>"<< x(0) << ";"<< x(1) << ";"<< x(2) << ";"<< x(3) << ";"<< x(4) << ";"<< x(5) << "</pose>" << std::endl;
        f << "<right> 0; 1; 0 </right>" << std::endl;
        f << "<down> 0; 0; 1 </down>" << std::endl;
        f << "<forward> 1; 0; 0</forward>" << std::endl;
        f << "<width>"<< cam.width() << "</width>" << std::endl;
        f << "<height>"<< cam.height() << "</height>" << std::endl;
        f << "<fx>"<< cam.K()(0,0) << "</fx>" << std::endl;
        f << "<cx>"<< cam.K()(0,2) << "</cx>" << std::endl;
        f << "<fy>"<< cam.K()(1,1) << "</fy>" << std::endl;
        f << "<cy>"<< cam.K()(1,2) << "</cy>" << std::endl;
        f << "<sx> 0 </sx>" << std::endl;
        f << "<kappa1>"<< cam._k1 << "</kappa1>" << std::endl;
        f << "<kappa2>"<< cam._k2 << "</kappa2>" << std::endl;
        f << "<kappa3>"<< cam._k3 << "</kappa3>" << std::endl;
        f << "<tau1>"<< cam._p1 << "</tau1>" << std::endl;
        f << "<tau2>"<< cam._p2 << "</tau2>" << std::endl;
        f << "</camera_model>" << std::endl;
        f.close();
    }else{
        std::cerr << "Unable to save camera file " << filename << std::endl;
    }
}

inline void SaveCamModelLeftRightVisionConvention(std::string filename_prefix, MatlabCamera cam, Eigen::Matrix4d T_lr)
{
    Eigen::Matrix3d RDFvision;RDFvision<< 1,0,0,  0,1,0,  0,0,1;
    Eigen::Matrix3d RDFrobot; RDFrobot << 0,1,0,  0,0, 1,  1,0,0;
    Eigen::Matrix4d T_vis_ro = Eigen::Matrix4d::Identity();
    T_vis_ro.block<3,3>(0,0) = RDFvision.transpose() * RDFrobot;
    Eigen::Matrix4d T_ro_vis = Eigen::Matrix4d::Identity();
    T_ro_vis.block<3,3>(0,0) = RDFrobot.transpose() * RDFvision;

    Eigen::Matrix4d Trobot_lr = T_ro_vis*T_lr*T_vis_ro;
    SaveCamModelRobotConvention(filename_prefix + "lcmod.xml", "left",  cam, Eigen::Matrix4d::Identity() );
    SaveCamModelRobotConvention(filename_prefix + "rcmod.xml", "right", cam, Trobot_lr );
}

#pragma once

#include <Eigen/Eigen>
#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

inline NppiRect GetTopLeftAlignedRegion(int w, int h, int blockx, int blocky)
{
    NppiRect ret;
    ret.width = blockx * (w / blockx);
    ret.height = blocky * (h / blocky);
    ret.x = 0;
    ret.y = 0;
    return ret;
}

inline NppiRect GetCenteredAlignedRegion(int w, int h, int blockx, int blocky)
{
    NppiRect ret;
    ret.width = blockx * (w / blockx);
    ret.height = blocky * (h / blocky);
    ret.x = (w - ret.width) / 2;
    ret.y = (h - ret.height) / 2;
    return ret;
}

inline int GetLevelFromMaxPixels(int w, int h, unsigned long maxpixels)
{
    int level = 0;
    while( (w >> level)*(h >> level) > maxpixels ) {
        ++level;
    }
    return level;
}

inline void CamModelScale(mvl::CameraModel& camModel, double scale)
{
    if(scale != 1.0) {
        mvl_camera_t* cam = camModel.GetModel();

        cam->linear.width  *= scale;
        cam->linear.height *= scale;

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

inline void CamModelScaleToDimensions(mvl::CameraModel& camModel, int w, int h)
{
    const double scale = w / camModel.Width();
    CamModelScale(camModel, scale);
}

inline void CamModelCropToRegionOfInterest(mvl::CameraModel& camModel, const NppiRect& roi)
{
    mvl_camera_t* cam = camModel.GetModel();
    cam->linear.cx -= roi.x;
    cam->linear.cy -= roi.y;
    cam->linear.width = roi.width;
    cam->linear.height = roi.height;
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

inline Eigen::Matrix3d MakeK(float fu, float fv, float u0, float v0)
{
    Eigen::Matrix3d K;
    K << fu, 0, u0,
         0, fv, v0,
         0,0,1;
    return K;
}

inline Eigen::Matrix3d MakeKinv(float fu, float fv, float u0, float v0)
{
    Eigen::Matrix3d K;
    K << 1.0/fu, 0, -u0/fu,
         0, 1.0/fv, -v0/fv,
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

class CameraModelPyramid : public mvl::CameraModel
{
public:
    inline CameraModelPyramid(const std::string& filename)
        : mvl::CameraModel(filename)
    {
        PopulatePyramid();
    }

    inline const Eigen::Matrix<double,3,3>& K(size_t i = 0) const
    {
        return m_K[i];
    }

    inline const Eigen::Matrix<double,3,3>& Kinv(size_t i = 0) const
    {
        return m_Kinv[i];
    }

    inline void PopulatePyramid(int max_levels = 10)
    {
        m_K.clear();
        m_Kinv.clear();
        unsigned level = 0;
        unsigned w = mvl::CameraModel::Width();
        unsigned h = mvl::CameraModel::Height();
        Eigen::Matrix3d K = mvl::CameraModel::K();

        while(level <= max_levels && w > 0 && h > 0)
        {
            m_K.push_back(K);
            m_Kinv.push_back(MakeKinv(K));
            level++;
            w = w/2;
            h = h/2;
            const Eigen::Matrix3d nk = ScaleK(K, 0.5);
            K = nk;
        }
    }

protected:
    std::vector<Eigen::Matrix3d> m_K;
    std::vector<Eigen::Matrix3d> m_Kinv;
};


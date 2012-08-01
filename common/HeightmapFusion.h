#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <Eigen/Eigen>
#include <Sophus/se3.h>

#include "../cu/Image.h"
#include "../cu/all.h"

class HeightmapFusion
{
public:
    HeightmapFusion(
        double HeightMapWidthMeters, double HeightMapHeightMeters,
        double PixelsPerMeter
    )
        : wm(HeightMapWidthMeters), hm(HeightMapHeightMeters),
          wp(wm*PixelsPerMeter), hp(hm*PixelsPerMeter),
          dHeightMap(wp, hp)
    {
        eT_hp << PixelsPerMeter, 0, 0, 0,
                 0, PixelsPerMeter, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1;
    }

    // Initialise with World to Plane coordinates.
    void Init(Eigen::Matrix4d T_pw)
    {
        eT_hw = eT_hp * T_pw;
        Gpu::InitHeightMap(dHeightMap);
    }

    void Fuse(Gpu::Image<float4> d3d, const Sophus::SE3& T_wc)
    {
        Eigen::Matrix<double,3,4> T_hc = (eT_hw * T_wc.matrix()).block<3,4>(0,0);
        Gpu::UpdateHeightMap(dHeightMap,d3d,Gpu::Image<unsigned char>(),T_hc);
    }

    void Fuse(Gpu::Image<float4> d3d, Gpu::Image<unsigned char> dImg, const Sophus::SE3& T_wc)
    {
        Eigen::Matrix<double,3,4> T_hc = (eT_hw * T_wc.matrix()).block<3,4>(0,0);
        Gpu::UpdateHeightMap(dHeightMap,d3d,dImg,T_hc);
    }

    void GenerateVbo(pangolin::GlBufferCudaPtr& vbo)
    {
        pangolin::CudaScopedMappedPtr var(vbo);
        Gpu::Image<float4> dVbo((float4*)*var,wp,hp);
        Gpu::VboFromHeightMap(dVbo,dHeightMap);
    }

    void GenerateCbo(pangolin::GlBufferCudaPtr& cbo)
    {
        pangolin::CudaScopedMappedPtr var(cbo);
        Gpu::Image<uchar4> dCbo((uchar4*)*var,wp,hp);
        Gpu::ColourHeightMap(dCbo,dHeightMap);
    }

    Gpu::Image<float4> GetHeightMap()
    {
        return dHeightMap;
    }

    Eigen::Matrix4d T_hw()
    {
        return eT_hw;
    }

    size_t WidthPixels() { return wp; }
    size_t HeightPixels() { return hp; }
    size_t WidthMeters() { return wm; }
    size_t HeightMeters() { return hm; }
    unsigned long Pixels() { return wp * hp; }

protected:
    // Width / Height in meters
    double wm;
    double hm;

    // Width / Height in pixels
    double wp;
    double hp;

    // Plane (z=0) to heightmap transform (adjust to pixel units)
    Eigen::Matrix4d eT_hp;

    // Heightmap to world transform (set once we know the plane)
    Eigen::Matrix4d eT_hw;

    Gpu::Image<float4,Gpu::TargetDevice,Gpu::Manage> dHeightMap;
};

#pragma once

#include <Eigen/Eigen>
#include <unsupported/Eigen/OpenGLSupport>
#include <SceneGraph/GLObject.h>
#include <SceneGraph/GLAxis.h>
#include <Mvlpp/SE3.h>
#include <iostream>
#include <GeographicLib/UTMUPS.hpp>

namespace SceneGraph
{

///////////////////////////////////////////////////////////////////////////////
// Definition
///////////////////////////////////////////////////////////////////////////////

class GLCameraHistory : public GLObject
{
public:
    GLCameraHistory();

    void LoadFromAbsoluteCartesianFile(
        const std::string& filename, int startframe, int endframe,
        const Eigen::Matrix4d T_hf, const Eigen::Matrix4d T_fh
    );

    void LoadFromTimeAbsoluteCartesianFile(
        const std::string& filename, int startframe, int endframe,
        const Eigen::Matrix4d T_hf, const Eigen::Matrix4d T_fh
    );

    void LoadFromRelativeCartesianFile(
        const std::string& filename, int startframe = 0
    );

    void LoadFromTimeRelativeCartesianFile(
        const std::string& filename, int startframe = 0
    );

    void LoadFromTimeLatLon( const std::string& filename);

    void DrawCanonicalObject();
    void SetNumberToShow(unsigned int n);

//protected:
    unsigned int m_numberToShow;

    // Chain of new to old relative transforms
    std::vector<Eigen::Matrix4d> m_T_on;

    // Absolute poses
    std::vector<Eigen::Matrix4d> m_T_wh;

    std::vector<double> m_time_s;

    // World to 'old' (inverse of last transform in vector)
    Eigen::Matrix4d T_ow;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

inline GLCameraHistory::GLCameraHistory()
    : m_numberToShow(1E6), T_ow(Eigen::Matrix4d::Identity())
{

}

inline  void GLCameraHistory::LoadFromAbsoluteCartesianFile(
    const std::string& filename, int startframe, int endframe,
    const Eigen::Matrix4d T_hf, const Eigen::Matrix4d T_fh
) {
    // Parse Ground truth
    std::ifstream ifs(filename);
    if(ifs.is_open()) {
        Eigen::Matrix<double,1,6> row;
        for(unsigned long lines = 0; ifs.good() && lines < endframe;lines++)
        {
            for(int i=0; i<6; ++i ) {
                ifs >> row(i);
            }
            if(lines >= startframe) {
                Eigen::Matrix4d T_wn( T_hf * mvl::Cart2T(row) * T_fh );
//                vecT_wh.push_back(T_wr);
                Eigen::Matrix4d T_on = T_ow * T_wn;
                m_T_on.push_back(T_on);
                T_ow = T_wn.inverse();
            }
        }

        ifs.close();
    }
}

inline  void GLCameraHistory::LoadFromTimeAbsoluteCartesianFile(
    const std::string& filename, int startframe, int endframe,
    const Eigen::Matrix4d T_hf, const Eigen::Matrix4d T_fh
) {
    // Parse Ground truth
    std::ifstream ifs(filename);
    if(ifs.is_open()) {
        Eigen::Matrix<double,1,6> row;
        for(unsigned long lines = 0; ifs.good() && lines < endframe;lines++)
        {
            double time_s = 0;
            ifs >> time_s;

            for(int i=0; i<6; ++i ) {
                ifs >> row(i);
            }

            if(lines >= startframe) {
                Eigen::Matrix4d T_wn( T_hf * mvl::Cart2T(row) * T_fh );
//                vecT_wh.push_back(T_wr);
                Eigen::Matrix4d T_on = T_ow * T_wn;
                m_T_on.push_back(T_on);
                T_ow = T_wn.inverse();
                m_time_s.push_back(time_s);
            }
        }

        ifs.close();
    }
}

inline void GLCameraHistory::LoadFromRelativeCartesianFile(
    const std::string& filename, int startframe
) {
    // Parse Ground truth
    std::ifstream ifs(filename);
    if(ifs.is_open()) {
        Eigen::Matrix<double,1,6> row;
        for(unsigned long lines = 0; ifs.good() && lines < 10000;lines++)
        {
            for(int i=0; i<6; ++i ) {
                ifs >> row(i);
            }
            if(lines >= startframe) {
                Eigen::Matrix4d T_on( mvl::Cart2T(row)  );
                m_T_on.push_back(T_on);
                T_ow = T_on.inverse() * T_ow;
            }
        }

        ifs.close();
    }
}

inline void GLCameraHistory::LoadFromTimeRelativeCartesianFile(
    const std::string& filename, int startframe
) {
    // Parse Ground truth
    std::ifstream ifs(filename);
    if(ifs.is_open()) {
        Eigen::Matrix<double,1,6> row;
        for(unsigned long lines = 0; ifs.good() && lines < 10000;lines++)
        {
            double time_s = 0;
            ifs >> time_s;

            for(int i=0; i<6; ++i ) {
                ifs >> row(i);
            }
            if(lines >= startframe) {
                Eigen::Matrix4d T_on( mvl::Cart2T(row)  );
                m_T_on.push_back(T_on);
                T_ow = T_on.inverse() * T_ow;
                m_time_s.push_back(time_s);
            }
        }

        ifs.close();
    }
}

inline void GLCameraHistory::LoadFromTimeLatLon( const std::string& filename)
{
    int _z = GeographicLib::UTMUPS::STANDARD;
    bool _np;
    double x_offset = 0;
    double y_offset = 0;

    // Parse Ground truth
    std::ifstream ifs(filename);
    if(ifs.is_open()) {
        Eigen::Matrix<double,1,6> row;
        for(unsigned long lines = 0; ifs.good() ;lines++)
        {
            double time_s = 0;
            double lat = 0;
            double lon = 0;
            double consume;

            ifs >> time_s;
            ifs >> lat;
            ifs >> lon;

            for(int i=0; i<4; ++i ) {
                ifs >> consume;
            }
            if(!ifs.fail()) {
                double x_meters;
                double y_meters;
                GeographicLib::UTMUPS::Forward(lat, lon, _z,_np,x_meters,y_meters,_z);

                if(m_T_on.size() == 0 ) {
                    x_offset = -x_meters;
                    y_offset = -y_meters;
                }

                Eigen::Vector6d cart;
                cart.setZero();
                cart(0) = x_meters + x_offset;
                cart(1) = y_meters + y_offset;

//                Eigen::Matrix4d T_on( mvl::Cart2T(cart)  );
//                m_T_on.push_back(T_on);


                Eigen::Matrix4d T_wn( mvl::Cart2T(cart) );
                Eigen::Matrix4d T_on = T_ow * T_wn;
                T_ow = T_wn.inverse();

                m_T_wh.push_back(T_wn);
                m_T_on.push_back(T_on);
                m_time_s.push_back(time_s);
            }
        }

        ifs.close();
    }
}

inline void GLCameraHistory::SetNumberToShow(unsigned int n)
{
    m_numberToShow = n;
}

inline void GLCameraHistory::DrawCanonicalObject()
{
    const int N = std::min((size_t)m_numberToShow, m_T_on.size());

    glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
    glDisable( GL_LIGHTING );
    glPushMatrix();

    for( unsigned int i=0; i < N; ++i )
    {
        glMultMatrix(m_T_on[i]);
        GLAxis::DrawUnitAxis();
    }

    glPopMatrix();
    glPopAttrib();
}


} // SceneGraph

#pragma once

#include <Eigen/Eigen>
#include <unsupported/Eigen/OpenGLSupport>
#include <SceneGraph/GLObject.h>
#include <SceneGraph/GLAxis.h>
#include <Mvlpp/SE3.h>

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
        const std::string& filename, int startframe,
        const Eigen::Matrix4d T_hf, const Eigen::Matrix4d T_fh
    );

    void LoadFromRelativeCartesianFile(
        const std::string& filename, int startframe
    );

    void DrawCanonicalObject();
    void SetNumberToShow(unsigned int n);

protected:
    unsigned int m_numberToShow;

    // Chain of new to old relative transforms
    std::vector<Eigen::Matrix4d> m_T_on;

    // World to 'old' (inverse of last transform in vector)
    Eigen::Matrix4d T_ow;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

inline GLCameraHistory::GLCameraHistory()
    : m_numberToShow(0), T_ow(Eigen::Matrix4d::Identity())
{

}

inline  void GLCameraHistory::LoadFromAbsoluteCartesianFile(
    const std::string& filename, int startframe,
    const Eigen::Matrix4d T_hf, const Eigen::Matrix4d T_fh
) {
    // Parse Ground truth
    std::ifstream ifs(filename);
    if(ifs.is_open()) {
        Eigen::Matrix<double,1,6> row;
        for(unsigned long lines = 0; lines < 10000;lines++)
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

void GLCameraHistory::LoadFromRelativeCartesianFile(
    const std::string& filename, int startframe
) {
    // Parse Ground truth
    std::ifstream ifs(filename);
    if(ifs.is_open()) {
        Eigen::Matrix<double,1,6> row;
        for(unsigned long lines = 0; lines < 10000;lines++)
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

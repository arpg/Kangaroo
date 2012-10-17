#pragma once

#include "PoseGraph.h"
#include <SceneGraph/SceneGraph.h>
#include <unsupported/Eigen/OpenGLSupport>

class GLPoseGraph : public SceneGraph::GLObject
{
public:
    GLPoseGraph(const PoseGraph& posegraph)
        : posegraph(posegraph)
    {
    }

    void DrawCanonicalObject()
    {
        glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
        glDisable( GL_LIGHTING );

        // Draw each keyframe
        const int N = posegraph.keyframes.size();
        const int M = posegraph.coord_frames.size();
        for(int i=0; i < N; ++i ) {
            const Keyframe& kf = posegraph.keyframes[i];
            glPushMatrix();
            Eigen::glMultMatrix(kf.GetT_wk().matrix());
            SceneGraph::GLAxis::DrawAxis(0.1);
            for(int j=0; j < M; ++j) {
                const Keyframe& cf = posegraph.coord_frames[j];
                glPushMatrix();
                Eigen::glMultMatrix(cf.GetT_wk().matrix());
                SceneGraph::GLAxis::DrawAxis(0.05);
                glPopMatrix();
            }
            glPopMatrix();
        }

        glPopAttrib();
    }

    const PoseGraph& posegraph;
};

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
        // Draw each keyframe
        const int N = posegraph.keyframes.size();
        for(int i=0; i < N; ++i ) {
            const Keyframe& kf = posegraph.keyframes[i];
            glPushMatrix();
            Eigen::glMultMatrix(kf.GetT_wk().matrix());
            SceneGraph::GLAxis::DrawAxis(0.1);
            glPopMatrix();
        }
    }

    const PoseGraph& posegraph;
};

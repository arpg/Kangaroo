#pragma once

#include <pangolin/pangolin.h>

inline pangolin::View& SetupPangoGL(int w, int h)
{
    // Setup OpenGL Display (based on GLUT)
    const int UI_WIDTH = 180;
    pangolin::CreateGlutWindowAndBind(__FILE__,UI_WIDTH+w,h);
    glewInit();

    // Setup default OpenGL parameters
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
    glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
    glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST );
    glEnable (GL_BLEND);
    glEnable (GL_LINE_SMOOTH);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(1.5);
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    // Tell the base view to arrange its children equally
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    pangolin::View& container = pangolin::CreateDisplay()
            .SetBounds(0,1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqual);

    return container;
}

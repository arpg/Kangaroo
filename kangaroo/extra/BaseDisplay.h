#pragma once

#include <pangolin/pangolin.h>

inline pangolin::View& SetupPangoGL(int w, int h, int ui_width = 180, std::string window_title = "-")
{
    // Setup OpenGL Display (based on GLUT)
    pangolin::CreateWindowAndBind(window_title,ui_width+w,h);
    if (glewInit() != GLEW_OK ) {
        std::cerr << "Unable to initialize GLEW." << std::endl;
    }

    // Setup default OpenGL parameters
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
    glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
    glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST );
    glEnable (GL_BLEND);
    glEnable (GL_LINE_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(1.5);
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    // Tell the base view to arrange its children equally
    if(ui_width != 0) {
        pangolin::CreatePanel("ui")
            .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(ui_width));
    }

    pangolin::View& container = pangolin::CreateDisplay()
            .SetBounds(0,1.0, pangolin::Attach::Pix(ui_width), 1.0);

    return container;
}

inline void SetupContainer(pangolin::View& container, int num_views, float aspect)
{
    container.SetLayout(pangolin::LayoutEqual);

    for(int i=0; i<num_views; ++i ) {
        pangolin::View& v = pangolin::CreateDisplay();
        v.SetAspect(aspect);
        container.AddDisplay(v);
    }

    pangolin::RegisterKeyPressCallback('~', [&container](){static bool showpanel=true; showpanel = !showpanel; if(showpanel) { container.SetBounds(0,1,pangolin::Attach::Pix(180), 1); }else{ container.SetBounds(0,1,0, 1); } pangolin::Display("ui").Show(showpanel); } );

    const int keys = 10;
    const char keyShowHide[] = {'1','2','3','4','5','6','7','8','9','0'};
    const char keySave[]     = {'!','@','#','$','%','^','&','*','(',')'};

    for(int v=0; v < num_views && v < keys; v++) {
        pangolin::RegisterKeyPressCallback(keyShowHide[v], [&container,v](){container[v].ToggleShow();} );
        pangolin::RegisterKeyPressCallback(keySave[v], [&container,v](){container[v].SaveRenderNow("screenshot",4);} );
    }
}

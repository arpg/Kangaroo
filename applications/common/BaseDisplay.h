#pragma once

#include <pangolin/pangolin.h>

inline pangolin::View& SetupPangoGL(int w, int h, int ui_width = 180)
{
    // Setup OpenGL Display (based on GLUT)
    pangolin::CreateGlutWindowAndBind(__FILE__,ui_width+w,h);
    glewInit();

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
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(ui_width));

    pangolin::View& container = pangolin::CreateDisplay()
            .SetBounds(0,1.0, pangolin::Attach::Pix(ui_width), 1.0)
            .SetLayout(pangolin::LayoutEqual);

    return container;
}

inline void SetupContainer(pangolin::View& container, int num_views, float aspect)
{
    for(int i=0; i<num_views; ++i ) {
        pangolin::View& v = pangolin::CreateDisplay();
        v.SetAspect(aspect);
        container.AddDisplay(v);
    }

    pangolin::RegisterKeyPressCallback('~', [&container](){static bool showpanel=true; showpanel = !showpanel; if(showpanel) { container.SetBounds(0,1,pangolin::Attach::Pix(180), 1); }else{ container.SetBounds(0,1,0, 1); } pangolin::Display("ui").Show(showpanel); } );

    pangolin::RegisterKeyPressCallback('1', [&container](){container[0].ToggleShow();} );
    pangolin::RegisterKeyPressCallback('2', [&container](){container[1].ToggleShow();} );
    pangolin::RegisterKeyPressCallback('3', [&container](){container[2].ToggleShow();} );
    pangolin::RegisterKeyPressCallback('4', [&container](){container[3].ToggleShow();} );
    pangolin::RegisterKeyPressCallback('5', [&container](){container[4].ToggleShow();} );
    pangolin::RegisterKeyPressCallback('6', [&container](){container[5].ToggleShow();} );

    pangolin::RegisterKeyPressCallback('!', [&container](){container[0].SaveRenderNow("screenshot",4);} );
    pangolin::RegisterKeyPressCallback('@', [&container](){container[1].SaveRenderNow("screenshot",4);} );
    pangolin::RegisterKeyPressCallback('#', [&container](){container[2].SaveRenderNow("screenshot",4);} );
    pangolin::RegisterKeyPressCallback('$', [&container](){container[3].SaveRenderNow("screenshot",4);} );
    pangolin::RegisterKeyPressCallback('%', [&container](){container[4].SaveRenderNow("screenshot",4);} );
    pangolin::RegisterKeyPressCallback('^', [&container](){container[5].SaveRenderNow("screenshot",4);} );
}

inline pangolin::View& SetupPangoGLWithCuda(int w, int h, int ui_width = 180, int min_gpu_mem_mb = 100 )
{
    pangolin::View& container = SetupPangoGL(w,h,ui_width);

    // Initialise CUDA, allowing it to use OpenGL context
    if( cudaGLSetGLDevice(0) != cudaSuccess ) {
        std::cerr << "Unable to get CUDA Device" << std::endl;
        exit(-1);
    }
    const unsigned bytes_per_mb = 1024*1000;
    size_t cu_mem_start, cu_mem_total;
    cudaMemGetInfo( &cu_mem_start, &cu_mem_total );
    std::cout << cu_mem_start/bytes_per_mb << " MB Video Memory Available." << std::endl;
    if( cu_mem_start < (min_gpu_mem_mb * bytes_per_mb) ) {
        std::cerr << "Not enough memory to proceed." << std::endl;
        exit(-1);
    }
    return container;
}

#include <SimpleGui/Gui.h>
#include <SimpleGui/GetPot>
#include <SimpleGui/GLMesh.h>

using namespace std;
using namespace Eigen;

const bool  USE_SCENE_GRAPH = false;
const char* MESH_NAME = "./Models/beatle-no-wheels-no-interior-embedded-texture.blend";

class Application
{
public:
    Application()
        : window(0, 0, 1024, 768, __FILE__ ),
          meshtest()
    {
        Init();
    }

    void Init()
    {
        meshtest.Init(MESH_NAME);

        if(USE_SCENE_GRAPH) {
            window.AddChildToRoot(&meshtest);
        }else{
            window.AddPostRenderCallback( Application::PostRender, this);
        }
    }

    int Run()
    {
        return window.Run();
    }

    void GlSanityTest()
    {
        glDisable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glClearColor (0.0, 0.0, 0.0, 0.0);
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glColor3f (1.0, 1.0, 1.0);
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
        glBegin(GL_POLYGON);
        glVertex3f (0.25, 0.25, 0.0);
        glVertex3f (0.75, 0.25, 0.0);
        glVertex3f (0.75, 0.75, 0.0);
        glVertex3f (0.25, 0.75, 0.0);
        glEnd();
    }

    static void PostRender(GLWindow*, void* data)
    {
        Application* self = (Application*)data;

//        self->GlSanityTest();

//        glScalef(0.03,0.03,0.03);
        self->meshtest.draw();
    }

    GLWindow window;
    GLMesh meshtest;
};

int main (int argc, char** argv){
    Application app;
    return app.Run();
}

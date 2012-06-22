#include <pangolin/pangolin.h>

using namespace std;
using namespace pangolin;

int main( int /*argc*/, char* argv[] )
{
    const int w = 640;
    const int h = 480;

    pangolin::CreateGlutWindowAndBind("Main",w,h);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        pangolin::FinishGlutFrame();
    }
}

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <npp.h>

#include <fiducials/drawing.h>

#include "common/RpgCameraOpen.h"
#include "common/DisplayUtils.h"
#include "common/BaseDisplay.h"
#include "common/ImageSelect.h"
#include "common/CameraModelPyramid.h"

#include <kangaroo/kangaroo.h>
#include <SceneGraph/SceneGraph.h>

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;
using namespace mvl;
using namespace SceneGraph;

int main( int argc, char* argv[] )
{
    CameraDevice video = OpenRpgCamera(argc,argv);

    std::vector<rpg::ImageWrapper> images;
    video.Capture(images);

    const unsigned int w = images[0].width();
    const unsigned int h = images[0].height();

    // Load Camera intrinsics from file
    CameraModel cam[] = {
        video.GetProperty("DataSourceDir") + "/lcmod.xml",
        video.GetProperty("DataSourceDir") + "/rcmod.xml"
    };

    const NppiRect roi = GetCenteredAlignedRegion(w,h,16,16);

    for(int i=0; i<2; ++i ) {
        CamModelScaleToDimensions(cam[i], w, h );
        CamModelCropToRegionOfInterest(cam[i], roi);
    }

    const Eigen::Matrix3d& K = cam[0].K();

    // Initialise window
    View& container = SetupPangoGL(1024, 768);
    SetupContainer(container, 4, (float)w/h);

    // Initialise CUDA, allowing it to use OpenGL context
    cudaGLSetGLDevice(0);

    Image<unsigned char, TargetDevice, Manage> img[] = {{w,h},{w,h}};
    Image<unsigned char, TargetDevice, Manage> temp(w,h);
    Image<float4, TargetDevice, Manage> debug[] = {{w,h},{w,h}};
    Image<unsigned char, TargetDevice,Manage> dScratch(w*sizeof(LeastSquaresSystem<float,3>),h);

    ActivateDrawImage<unsigned char> adImgL(img[0],GL_LUMINANCE8, false, true);
    ActivateDrawImage<unsigned char> adImgR(img[1],GL_LUMINANCE8, false, true);
    ActivateDrawImage<float4> adDebug1(debug[1],GL_RGBA_FLOAT32_APPLE, false, true);

    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,K(0,0),K(1,1),K(0,2),K(1,2),0.1,10000),
        IdentityMatrix(GlModelViewStack)
    );

    GLSceneGraph glgraph;
    GLAxis glaxis;
    glgraph.AddChild(&glaxis);

    container[0].SetDrawFunction(boost::ref(adImgL));
    container[1].SetDrawFunction(boost::ref(adImgR));
    container[2].SetDrawFunction(ActivateDrawFunctor(glgraph,s_cam));
    container[3].SetDrawFunction(boost::ref(adDebug1));

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", false, true);

    Var<int> blur("ui.blur",3,0,10);
    Var<bool> opt("ui.optimise", false);
    Var<float> cut("ui.cut",0.1,0,0.2);
    Var<float> min_grad("ui.min grad",0.01,0,0.02);
    Var<float> scale("ui.scale",100,1,1E4);

    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    Sophus::SO3d R;

    for(unsigned long frame=0; !pangolin::ShouldQuit(); )
    {
        const bool go = (frame==0) || run || Pushed(step);
        const bool guichanged = GuiVarHasChanged();

        if(go) {
            video.Capture(images);
            frame++;
            for(int i=0; i<2; ++i ) {
                img[i].MemcpyFromHost(images[i].Image.data, w);
                for(int b=0; b<blur; ++b) {
                    Blur(img[i],temp);
                }
            }
        }

//        if(go || guichanged )
        {
            for(int i=0; i<5; ++i )
            {
                LeastSquaresSystem<float,3> lss = ManhattenLineCost(
                    debug[0], debug[1], img[0], R.matrix(),
                    K(0,0), K(1,1), K(0,2), K(1,2), cut, scale, min_grad,
                    dScratch
                );
                Eigen::FullPivLU<Eigen::Matrix3d> lu_JTJ( (Eigen::Matrix3d)lss.JTJ );
                Eigen::Vector3d x = 1.0 * lu_JTJ.solve( (Eigen::Vector3d)lss.JTy );

                if(opt) {
                    R = R * Sophus::SO3::exp(x);
                }
            }
        glaxis.SetPose(Sophus::SE3d(R.inverse(),Eigen::Vector3d(0,0,2)).matrix());
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        pangolin::FinishGlutFrame();
    }
}

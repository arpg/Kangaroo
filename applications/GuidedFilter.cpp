#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/glsl.h>
#include <npp.h>

#include "common/RpgCameraOpen.h"
#include "common/DisplayUtils.h"
#include "common/BaseDisplayCuda.h"

#include <kangaroo/kangaroo.h>

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;

int main( int argc, char* argv[] )
{
    // Open video device
    CameraDevice video = OpenRpgCamera(argc,argv,1);

    // Capture first image
    std::vector<rpg::ImageWrapper> images;
    video.Capture(images);

    // Image dimensions
    const unsigned int w = images[0].width();
    const unsigned int h = images[0].height();

    cout << "Image has dimensions: " << w << "x" << h << endl;

    // Initialise window
    View& container = SetupPangoGLWithCuda(1024, 768);
    SetupContainer(container, 2, (float)w/h);

    // Texture we will use to display camera images
    GlTextureCudaArray tex8(w,h,GL_LUMINANCE8);
    GlTextureCudaArray texf(w,h,GL_LUMINANCE32F_ARB);

    // Allocate Camera Images on device for processing
    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> img(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> I(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> P(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> II(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> IP(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> meanI(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> meanP(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> meanII(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> meanIP(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> covIP(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> varI(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> a(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> b(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> meana(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> meanb(w,h);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage> q(w,h);

    Gpu::Image<unsigned char, Gpu::TargetDevice, Gpu::Manage> Scratch(w*sizeof(int),h);

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", true, true);
    Var<float> eps("ui.eps",0.1, 0, 0.5);
    Var<int> rad("ui.radius",10, 1, 20);

    for(unsigned long frame=0; !pangolin::ShouldQuit() /*&& frame < 100*/; ++frame)
    {
        const bool go = (frame==0) || run || Pushed(step);

        if(go) {
            video.Capture(images);
            img.MemcpyFromHost(images[0].Image.data );
            Gpu::ElementwiseScaleBias<float,unsigned char,float>(I,img,1/255.0,0);
            P.CopyFrom(I);
        }

        if(go || GuiVarHasChanged() ) {
            // Guided Image Filtering (ECCV 2010)
            // Kaiming He, Jian Sun, and Xiaoou Tang

            // mean_I = boxfilter(I, r) ./ N;
            Gpu::BoxFilter<float,float,float>(meanI,I,Scratch,rad);

            // mean_II = boxfilter(I.*I, r) ./ N;
            Gpu::ElementwiseSquare<float,float,float>(II,I);
            Gpu::BoxFilter<float,float,float>(meanII,II,Scratch,rad);

            // mean_p = boxfilter(p, r) ./ N;
            Gpu::BoxFilter<float,float,float>(meanP,P,Scratch,rad);

            // mean_Ip = boxfilter(I.*p, r) ./ N;
            Gpu::ElementwiseMultiply<float,float,float,float>(IP,I,P);
            Gpu::BoxFilter<float,float,float>(meanIP,IP,Scratch,rad);

            // cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
            Gpu::ElementwiseMultiplyAdd<float,float,float,float,float>(covIP, meanI, meanP, meanIP, -1);

            // var_I = mean_II - mean_I .* mean_I;
            Gpu::ElementwiseMultiplyAdd<float,float,float,float,float>(varI, meanI, meanI, meanII,-1);

            // a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
            Gpu::ElementwiseDivision<float,float,float,float>(a, covIP, varI, 0, eps);

            // b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
            Gpu::ElementwiseMultiplyAdd<float,float,float,float,float>(b, a,meanI,meanP,-1);

            // mean_a = boxfilter(a, r) ./ N;
            Gpu::BoxFilter<float,float,float>(meana,a,Scratch,rad);

            // mean_b = boxfilter(b, r) ./ N;
            Gpu::BoxFilter<float,float,float>(meanb,b,Scratch,rad);

            // q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
            Gpu::ElementwiseMultiplyAdd<float,float,float,float,float>(q,meana,I,meanb);
        }

        /////////////////////////////////////////////////////////////
        // Perform drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        if(container[0].IsShown()) {
            container[0].Activate();
            tex8 << img;
            tex8.RenderToViewportFlipY();
        }

        if(container[1].IsShown()) {
            container[1].Activate();
            texf << q;
//            GlSlUtilities::Scale(1/255.0f);
            texf.RenderToViewportFlipY();
//            GlSlUtilities::UseNone();
        }

        pangolin::FinishGlutFrame();
    }
}

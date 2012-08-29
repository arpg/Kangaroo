#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <Pangolin/glsl.h>
#include <npp.h>

#include <SceneGraph/SceneGraph.h>
#include <SceneGraph/GLVbo.h>
#include "common/GLCameraHistory.h"

#include <fiducials/drawing.h>
#include <fiducials/camera.h>

#include "common/RpgCameraOpen.h"
#include "common/DisplayUtils.h"
#include "common/ScanlineRectify.h"
#include "common/ImageSelect.h"
#include "common/BaseDisplay.h"
#include "common/HeightmapFusion.h"
#include "common/CameraModelPyramid.h"
#include "common/LoadPosesFromFile.h"

#include <kangaroo/kangaroo.h>

#include <Node.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;

int main( int argc, char* argv[] )
{
    // Open video device
    CameraDevice video = OpenRpgCamera(argc,argv);

    // Capture first image
    std::vector<rpg::ImageWrapper> img;
    video.Capture(img);

    // native width and height (from camera)
    const unsigned int w = img[0].width();
    const unsigned int h = img[0].height();

    // Initialise window
    View& container = SetupPangoGL(1024, 768);
    SetupContainer(container, 4, (float)w/h);

    // Initialise CUDA, allowing it to use OpenGL context
    cudaGLSetGLDevice(0);

    // Allocate Camera Images on device for processing
    Image<unsigned char, TargetDevice, Manage> CamImg[] = {{w,h},{w,h}};
    Image<unsigned char, TargetDevice, Manage> Img[] = {{w,h},{w,h}};

    Image<char, TargetDevice, Manage> DispInt[] = {{w,h},{w,h}};
    Image<float, TargetDevice, Manage>  Disp[] = {{w,h},{w,h}};

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", false, true);

    Var<float> maxPosDisp("ui.disp",80, 0, 126);
    Var<float> dispStep("ui.disp step",1, 0.1, 1);
    Var<int> scoreRad("ui.score rad",1, 0, 5 );
    Var<bool> scoreNormed("ui.score normed",true, true);

    Var<float> stereoAcceptThresh("ui.2nd Best thresh", 0, 0, 1, false);

    Var<bool> subpix("ui.subpix", false, true);
    Var<bool> reverse_check("ui.reverse_check", false, true);

    Var<bool> applyBilateralFilter("ui.Apply Bilateral Filter", false, true);
    Var<int> bilateralWinSize("ui.size",5, 1, 20);
    Var<float> gs("ui.gs",2, 1E-3, 5);
    Var<float> gr("ui.gr",0.0184, 1E-3, 1);

    Var<int> domedits("ui.median its",1, 1, 10);
    Var<bool> domed9x9("ui.median 9x9", false, true);
    Var<bool> domed7x7("ui.median 7x7", false, true);
    Var<bool> domed5x5("ui.median 5x5", false, true);
    Var<bool> domed3x3("ui.median 3x3", false, true);
    Var<int> medi("ui.medi",12, 0, 24);

    Var<float> filtgradthresh("ui.filt grad thresh", 0, 0, 20);

    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    ActivateDrawImage<unsigned char> adImgL(CamImg[0],GL_LUMINANCE8, false, true);
    ActivateDrawImage<unsigned char> adImgR(CamImg[1],GL_LUMINANCE8, false, true);

    ActivateDrawImage<float> adDispL(Disp[0],GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adDispR(Disp[1],GL_LUMINANCE32F_ARB, false, true);

    container[0].SetDrawFunction(boost::ref(adImgL));
    container[1].SetDrawFunction(boost::ref(adImgR));
    container[2].SetDrawFunction(boost::ref(adDispL));
    container[3].SetDrawFunction(boost::ref(adDispR));

    for(unsigned long frame=0; !pangolin::ShouldQuit();)
    {
        const bool go = frame==0 || run || Pushed(step);

        if(go) {
            video.Capture(img);
            frame++;

            for(int i=0; i<2; ++i ) {
                CamImg[i].MemcpyFromHost(img[i].Image.data, w);
            }
        }

        if(go || GuiVarHasChanged() ) {
            for(int i=0; i<2; ++i) {
                const size_t img1 = i;
                const size_t img2 = 1-i;
                const char maxDisp = -(2*i-1) * maxPosDisp;

                DenseStereo<char,unsigned char>(DispInt[img1], CamImg[img1], CamImg[img2], maxDisp, stereoAcceptThresh, scoreRad);

//                if(dispStep == 1 )
//                {
//                    // Compute dense stereo
//                    DenseStereo(DispInt[img1], CamImg[img1], CamImg[img2], maxDisp, stereoAcceptThresh, scoreRad);

//                    if(reverse_check) {
//                        ReverseCheck(DispInt[img1], CamImg[img1], CamImg[img2] );
//                    }

//                    if(subpix) {
//                        DenseStereoSubpixelRefine(Disp[img1], DispInt[img1], CamImg[img1], CamImg[img2]);
//                    }else{
//                        ConvertImage<float, unsigned char>(Disp[img1], DispInt[img1]);
//                    }
//                }else{
//                    DenseStereo(Disp[img1], CamImg[img1], CamImg[img2], maxDisp, dispStep, stereoAcceptThresh, scoreRad, scoreNormed);
//                }

                ConvertImage<float, char>(Disp[img1], DispInt[img1]);
                nppiDivC_32f_C1IR(maxDisp,Disp[img1].ptr,Disp[img1].pitch,Disp[img1].Size());
            }


//            for(int i=0; i < domedits; ++i ) {
//                if(domed9x9) MedianFilterRejectNegative9x9(DispL,DispL, medi);
//                if(domed7x7) MedianFilterRejectNegative7x7(DispL,DispL, medi);
//                if(domed5x5) MedianFilterRejectNegative5x5(DispL,DispL, medi);
//                if(domed3x3) MedianFilter3x3(DispL,DispL);
//            }

//            if(filtgradthresh > 0) {
//                FilterDispGrad(DispL, DispL, filtgradthresh);
//            }

//            if(applyBilateralFilter) {
//                BilateralFilter(DispFilt,DispL,gs,gr,bilateralWinSize);
//                DispL.CopyFrom(DispFilt);
//            }

            // normalise dDisp
//            nppiDivC_32f_C1IR(maxDisp,DispL.ptr,DispL.pitch,DispL.Size());
        }

        /////////////////////////////////////////////////////////////
        // Draw

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

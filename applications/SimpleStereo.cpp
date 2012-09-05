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
    std::vector<rpg::ImageWrapper> images;
    video.Capture(images);

    // native width and height (from camera)
    const unsigned int w = images[0].width();
    const unsigned int h = images[0].height();
    const unsigned int MAXD = 256;

    // Initialise window
    View& container = SetupPangoGL(1024, 768);
    SetupContainer(container, 4, (float)w/h);

    // Initialise CUDA, allowing it to use OpenGL context
    cudaGLSetGLDevice(0);

    // Allocate Camera Images on device for processing
    Image<unsigned char, TargetDevice, Manage> img[] = {{w,h},{w,h}};
    Volume<unsigned short, TargetDevice, Manage> vol(w,h,MAXD);
    Volume<float, TargetDevice, Manage> sgm(w,h,MAXD);

//    Image<unsigned long, TargetDevice, Manage> census[] = {{w,h},{w,h}};
    Image<ulong2, TargetDevice, Manage> census[] = {{w,h},{w,h}};
    Image<float, TargetDevice, Manage>  Disp[] = {{w,h},{w,h}};

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", true, true);

    Var<int> maxPosDisp("ui.disp",80, 0, MAXD-1);
    Var<int> scoreRad("ui.score rad",4, 0, 7 );
//    Var<float> dispStep("ui.disp step",1, 0.1, 1);
//    Var<bool> scoreNormed("ui.score normed",true, true);

    Var<float> stereoAcceptThresh("ui.2nd Best thresh", 0, 0, 1, false);

    Var<int> reverse_check("ui.reverse_check", -1, -1, 5);
//    Var<bool> subpix("ui.subpix", false, true);

//    Var<bool> applyBilateralFilter("ui.Apply Bilateral Filter", false, true);
//    Var<int> bilateralWinSize("ui.size",5, 1, 20);
//    Var<float> gs("ui.gs",2, 1E-3, 5);
//    Var<float> gr("ui.gr",0.0184, 1E-3, 1);

    Var<int> domedits("ui.median its",1, 1, 10);
    Var<bool> domed9x9("ui.median 9x9", false, true);
    Var<bool> domed7x7("ui.median 7x7", false, true);
    Var<bool> domed5x5("ui.median 5x5", false, true);
    Var<bool> domed3x3("ui.median 3x3", false, true);
    Var<int> medi("ui.medi",12, 0, 24);

    Var<bool> dosgm("ui.sgm", true, true);
    Var<float> sgmP1("ui.P1",1, 0, 100);
    Var<float> sgmP2("ui.P2",500, 0, 1000);
    Var<bool> dohoriz("ui.horiz", true, true);
    Var<bool> dovert("ui.vert", true, true);
    Var<bool> doreverse("ui.reverse", false, true);

//    Var<float> filtgradthresh("ui.filt grad thresh", 0, 0, 20);
//    Var<float> sigma("ui.sigma", 1, 0, 20);

    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    ActivateDrawImage<unsigned char> adImgL(img[0],GL_LUMINANCE8, false, true);
    ActivateDrawImage<unsigned char> adImgR(img[1],GL_LUMINANCE8, false, true);

    ActivateDrawImage<float> adDispL(Disp[0],GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adDispR(Disp[1],GL_LUMINANCE32F_ARB, false, true);

    container[0].SetDrawFunction(boost::ref(adImgL));
    container[1].SetDrawFunction(boost::ref(adImgR));
    container[2].SetDrawFunction(boost::ref(adDispL));
    container[3].SetDrawFunction(boost::ref(adDispR));

    CudaTimer cutimer;

    for(unsigned long frame=0; !pangolin::ShouldQuit() /*&& frame < 10*/;)
    {
        const bool go = frame==0 || run || Pushed(step);
        const bool guichanged = GuiVarHasChanged();

        if(guichanged) {
            cutimer.Reset();
        }

        if(go) {
            video.Capture(images);
            frame++;

            for(int i=0; i<2; ++i ) {
                img[i].MemcpyFromHost(images[i].Image.data, w);
            }
        }

        if(go || guichanged ) {

            for(int i=0; i<2; ++i) {
                Census(census[i], img[i]);
            }

            CensusStereoVolume(vol, census[0], census[1], maxPosDisp);
            if(dosgm) {
                SemiGlobalMatching(sgm,vol,img[0],maxPosDisp,sgmP1,sgmP2,dohoriz,dovert,doreverse);
                CostVolMinimum<float,float>(Disp[0],sgm,maxPosDisp);
            }else{
                CostVolMinimum<float,unsigned short>(Disp[0],vol,maxPosDisp);
            }
            nppiDivC_32f_C1IR(maxPosDisp,Disp[0].ptr,Disp[0].pitch,Disp[0].Size());

//            for(int i=0; i<2; ++i) {
//                const size_t img1 = i;
//                const size_t img2 = 1-i;
//                const char maxDisp = -(2*i-1) * maxPosDisp;
//                CensusStereo(DispInt[img1], census[img1], census[img2], maxDisp);
//            }

//            if(reverse_check >= 0) {
//                LeftRightCheck(DispInt[0], DispInt[1], reverse_check);
//            }

//            DenseStereo<char,unsigned char>(DispInt[1], img[0], img[1], maxPosDisp, 0, scoreRad);

//            for(int i=0; i<1; ++i) {
////                ConvertImage<float, char>(Disp[i], DispInt[i]);
//                const char maxDisp = maxPosDisp; //-(2*i-1) * maxPosDisp;
//                nppiDivC_32f_C1IR(maxDisp,Disp[i].ptr,Disp[i].pitch,Disp[i].Size());
//            }

//            DenseStereo<char,unsigned char>(DispInt[1], img[0], img[1], maxPosDisp, 0, scoreRad);
//            ConvertImage<float, char>(Disp[1], DispInt[1]);

//            nppiDivC_32f_C1IR(maxPosDisp,Disp[0].ptr,Disp[0].pitch,Disp[0].Size());
//            nppiDivC_32f_C1IR(maxPosDisp,Disp[1].ptr,Disp[1].pitch,Disp[1].Size());


            for(int i=0; i < domedits; ++i ) {
                if(domed9x9) MedianFilterRejectNegative9x9(Disp[0],Disp[0], medi);
                if(domed7x7) MedianFilterRejectNegative7x7(Disp[0],Disp[0], medi);
                if(domed5x5) MedianFilterRejectNegative5x5(Disp[0],Disp[0], medi);
                if(domed3x3) MedianFilter3x3(Disp[0],Disp[0]);
            }

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

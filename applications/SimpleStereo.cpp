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
    cout << images[0].width() << "x" << images[1].height() << endl;

    // native width and height (from camera)
    const unsigned int w = images[0].width();
    const unsigned int h = images[0].height();
    const unsigned int MAXD = 80;

    // Initialise window
    View& container = SetupPangoGLWithCuda(1024, 768);
    SetupContainer(container, 4, (float)w/h);

    // Allocate Camera Images on device for processing
    Image<unsigned char, TargetDevice, Manage> upload(w,h);
    Image<float, TargetDevice, Manage> img[] = {{w,h},{w,h}};
    Volume<float, TargetDevice, Manage> vol[] = {{w,h,MAXD},{w,h,MAXD},{w,h,MAXD}};
    Image<float, TargetDevice, Manage>  disp[] = {{w,h},{w,h}};

//    Image<float, TargetDevice, Manage> I(w,h);
//    Image<float, TargetDevice, Manage> P(w,h);
    Image<float, TargetDevice, Manage> II(w,h);
    Image<float, TargetDevice, Manage> IP(w,h);
    Image<float, TargetDevice, Manage> meanI(w,h);
    Image<float, TargetDevice, Manage> meanP(w,h);
    Image<float, TargetDevice, Manage> meanII(w,h);
    Image<float, TargetDevice, Manage> meanIP(w,h);
    Image<float, TargetDevice, Manage> covIP(w,h);
    Image<float, TargetDevice, Manage> varI(w,h);
    Image<float, TargetDevice, Manage> a(w,h);
    Image<float, TargetDevice, Manage> b(w,h);
    Image<float, TargetDevice, Manage> meana(w,h);
    Image<float, TargetDevice, Manage> meanb(w,h);
    Image<float, TargetDevice, Manage> temp(w,h);
    Image<unsigned char, TargetDevice, Manage> Scratch(w*sizeof(float),h);

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", true, true);
    Var<int> maxdisp("ui.maxdisp",MAXD, 0, MAXD);
    Var<bool> subpix("ui.subpix", true, true);
    Var<bool> filter("ui.filter", true, true);

    Var<float> alpha("ui.alpha", 0.9, 0,1);
    Var<float> r1("ui.r1", 0.0028, 0,0.01);
    Var<float> r2("ui.r2", 0.008, 0,0.01);

    Var<float> eps("ui.eps",0.01*0.01, 0, 0.01);
    Var<int> rad("ui.radius",9, 1, 20);

//    Var<int> showd("ui.showd",10, 0, MAXD-1);
//    Var<float> scale("ui.scale",1, 0.1, 100);

    Var<bool> leftrightcheck("ui.left-right check", true, true);
    Var<int> maxdispdiff("ui.maxdispdiff",1, 0, 5);

    Var<bool> smooth("ui.smooth", false, true);
    Var<float> eps2("ui.eps2",0.01*0.01, 0, 0.01);
    Var<int> rad2("ui.radius2",9, 1, 20);

    Var<bool> applyBilateralFilter("ui.Apply Bilateral Filter", false, true);
    Var<int> bilateralWinSize("ui.size",5, 1, 20);
    Var<float> gs("ui.gs",5, 1E-3, 10);
    Var<float> gr("ui.gr",6, 1E-3, 10);
    Var<float> gc("ui.gc",0.01, 1E-4, 0.1);

    Var<int> domedits("ui.median its",1, 1, 10);
    Var<bool> domed9x9("ui.median 9x9", false, true);
    Var<bool> domed7x7("ui.median 7x7", false, true);
    Var<bool> domed5x5("ui.median 5x5", false, true);
//    Var<bool> domed3x3("ui.median 3x3", false, true);
    Var<int> medi("ui.medi",12, 0, 24);

    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );

    ActivateDrawImage<float> adImgL(img[0],GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adImgR(img[1],GL_LUMINANCE32F_ARB, false, true);

    ActivateDrawImage<float> adDispL(disp[0],GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adDispR(disp[1],GL_LUMINANCE32F_ARB, false, true);
//    adDispL.SetImageScale(1.0f/MAXD);
//    adDispR.SetImageScale(1.0f/MAXD);

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
                Image<unsigned char,TargetHost> himg(images[i].Image.data, images[i].width(),images[i].height());
                upload.CopyFrom(himg.SubImage(w,h));
                upload.MemcpyFromHost(images[i].Image.data, w);
                ElementwiseScaleBias<float,unsigned char>(img[i], upload,1.0f/255.0f);
            }
        }

        if(go || guichanged ) {
            CostVolumeFromStereoTruncatedAbsAndGrad(vol[0], img[0], img[1], -1, alpha, r1, r2);
            CostVolumeFromStereoTruncatedAbsAndGrad(vol[1], img[1], img[0], +1, alpha, r1, r2);

            if(filter) {
                // Filter Cost volume
                for(int v=0; v<2; ++v)
                {
                    Image<float, TargetDevice, Manage>& I = img[v];

                    // Filter guide image
                    {
                        // mean_I = boxfilter(I, r) ./ N;
                        BoxFilter<float,float,float>(meanI,I,Scratch,rad);

                        // mean_II = boxfilter(I.*I, r) ./ N;
                        ElementwiseSquare<float,float,float>(II,I);
                        BoxFilter<float,float,float>(meanII,II,Scratch,rad);

                        // var_I = mean_II - mean_I .* mean_I;
                        ElementwiseMultiplyAdd<float,float,float,float,float>(varI, meanI, meanI, meanII,-1);
                    }

                    for(int d=0; d<maxdisp; ++d)
                    {
                        Image<float> P = vol[v].ImageXY(d);
                        Image<float> q = vol[v].ImageXY(d);

                        // mean_p = boxfilter(p, r) ./ N;
                        BoxFilter<float,float,float>(meanP,P,Scratch,rad);

                        // mean_Ip = boxfilter(I.*p, r) ./ N;
                        ElementwiseMultiply<float,float,float,float>(IP,I,P);
                        BoxFilter<float,float,float>(meanIP,IP,Scratch,rad);

                        // cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
                        ElementwiseMultiplyAdd<float,float,float,float,float>(covIP, meanI, meanP, meanIP, -1);

                        // a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
                        ElementwiseDivision<float,float,float,float>(a, covIP, varI, 0, eps);

                        // b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
                        ElementwiseMultiplyAdd<float,float,float,float,float>(b, a,meanI,meanP,-1);

                        // mean_a = boxfilter(a, r) ./ N;
                        BoxFilter<float,float,float>(meana,a,Scratch,rad);

                        // mean_b = boxfilter(b, r) ./ N;
                        BoxFilter<float,float,float>(meanb,b,Scratch,rad);

                        // q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
                        ElementwiseMultiplyAdd<float,float,float,float,float>(q,meana,I,meanb);
                    }
                }
            }

            if(subpix) {
                CostVolMinimumSubpix(disp[0],vol[0], maxdisp);
                CostVolMinimumSubpix(disp[1],vol[1], maxdisp);
            }else{
                CostVolMinimum<float,float>(disp[0],vol[0], maxdisp);
                CostVolMinimum<float,float>(disp[1],vol[1], maxdisp);
            }

            if(filter && smooth) {
                Image<float> I = img[0];
                Image<float> P = disp[0];
                Image<float> q = disp[0];

                // mean_p = boxfilter(p, r) ./ N;
                BoxFilter<float,float,float>(meanP,P,Scratch,rad2);

                // mean_Ip = boxfilter(I.*p, r) ./ N;
                ElementwiseMultiply<float,float,float,float>(IP,I,P);
                BoxFilter<float,float,float>(meanIP,IP,Scratch,rad2);

                // cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
                ElementwiseMultiplyAdd<float,float,float,float,float>(covIP, meanI, meanP, meanIP, -1);

                // a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
                ElementwiseDivision<float,float,float,float>(a, covIP, varI, 0, eps2);

                // b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
                ElementwiseMultiplyAdd<float,float,float,float,float>(b, a,meanI,meanP,-1);

                // mean_a = boxfilter(a, r) ./ N;
                BoxFilter<float,float,float>(meana,a,Scratch,rad2);

                // mean_b = boxfilter(b, r) ./ N;
                BoxFilter<float,float,float>(meanb,b,Scratch,rad2);

                // q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
                ElementwiseMultiplyAdd<float,float,float,float,float>(q,meana,I,meanb);
            }

            for(int di=0; di<2; ++di) {
                for(int i=0; i < domedits; ++i ) {
                    if(domed9x9) MedianFilterRejectNegative9x9(disp[di],disp[di], medi);
                    if(domed7x7) MedianFilterRejectNegative7x7(disp[di],disp[di], medi);
                    if(domed5x5) MedianFilterRejectNegative5x5(disp[di],disp[di], medi);
                }
            }

            if(applyBilateralFilter) {
                Image<float> I = img[0];
                Image<float> P = disp[0];
                temp.CopyFrom(P);
                BilateralFilter<float,float,float>(P,temp,I,gs,gr,gc,bilateralWinSize);
            }

            if(leftrightcheck ) {
                LeftRightCheck(disp[1], disp[0], +1, maxdispdiff);
                LeftRightCheck(disp[0], disp[1], -1, maxdispdiff);
            }

            ElementwiseScaleBias<float,float,float>(disp[0],disp[0],1.0f/maxdisp);
            ElementwiseScaleBias<float,float,float>(disp[1],disp[1],1.0f/maxdisp);

        }

//        adDispL.SetImageScale(scale);
//        adDispR.SetImageScale(scale);
//        adDispL.SetImage(vol[0].ImageXY(showd));
//        adDispR.SetImage(vol[0].ImageXY(showd));

        /////////////////////////////////////////////////////////////
        // Draw

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

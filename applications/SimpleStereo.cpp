#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/glsl.h>
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
#include "common/BaseDisplayCuda.h"
#include "common/HeightmapFusion.h"
#include "common/CameraModelPyramid.h"
#include "common/LoadPosesFromFile.h"
#include "common/SavePPM.h"

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>


const int MAXD = 80;

using namespace std;
using namespace pangolin;
using namespace Gpu;

int main( int argc, char* argv[] )
{
    // Initialise window
    View& container = SetupPangoGLWithCuda(1024, 768);
    size_t cu_mem_start, cu_mem_end, cu_mem_total;
    cudaMemGetInfo( &cu_mem_start, &cu_mem_total );
    glClearColor(1,1,1,0);

    // Open video device
    CameraDevice video = OpenRpgCamera(argc,argv,2,true);

    // Capture first image
    std::vector<rpg::ImageWrapper> images;
    video.Capture(images);

    // native width and height (from camera)
    const unsigned int nw = images[0].width();
    const unsigned int nh = images[0].height();

    // Downsample this image to process less pixels
    const int max_levels = 6;
    const int level = GetLevelFromMaxPixels( nw, nh, 640*480 );
//    const int level = 4;
    assert(level <= max_levels);

    // Find centered image crop which aligns to 16 pixels at given level
    const NppiRect roi = GetCenteredAlignedRegion(nw,nh,16 << level,16 << level);

    // Load Camera intrinsics from file
    CameraModelPyramid cam[] = {
        video.GetProperty("DataSourceDir") + "/lcmod.xml",
        video.GetProperty("DataSourceDir") + "/rcmod.xml"
    };

    for(int i=0; i<2; ++i ) {
        // Adjust to match camera image dimensions
        CamModelScaleToDimensions(cam[i], nw, nh );

        // Adjust to match cropped aligned image
        CamModelCropToRegionOfInterest(cam[i], roi);

        cam[i].PopulatePyramid(max_levels);
    }

    const unsigned int w = roi.width;
    const unsigned int h = roi.height;
    const unsigned int lw = w >> level;
    const unsigned int lh = h >> level;

    const Eigen::Matrix3d& K0 = cam[0].K();
    const Eigen::Matrix3d& Kl = cam[0].K(level);

    cout << "Video stream dimensions: " << nw << "x" << nh << endl;
    cout << "Chosen Level: " << level << endl;
    cout << "Processing dimensions: " << lw << "x" << lh << endl;
    cout << "Offset: " << roi.x << "x" << roi.y << endl;

    Eigen::Matrix3d RDFvision;RDFvision<< 1,0,0,  0,1,0,  0,0,1;
    Eigen::Matrix3d RDFrobot; RDFrobot << 0,1,0,  0,0, 1,  1,0,0;
    Eigen::Matrix4d T_vis_ro = Eigen::Matrix4d::Identity();
    T_vis_ro.block<3,3>(0,0) = RDFvision.transpose() * RDFrobot;
    Eigen::Matrix4d T_ro_vis = Eigen::Matrix4d::Identity();
    T_ro_vis.block<3,3>(0,0) = RDFrobot.transpose() * RDFvision;

    const Sophus::SE3 T_rl_orig = T_rlFromCamModelRDF(cam[0], cam[1], RDFvision);
    double k1 = 0;
    double k2 = 0;

    if(cam[0].Type() == MVL_CAMERA_WARPED)
    {
        k1 = cam[0].GetModel()->warped.kappa1;
        k2 = cam[0].GetModel()->warped.kappa2;
    }

    const bool rectify = (k1!=0 || k2!=0); // || camModel[0].GetPose().block<3,3>(0,0)
    if(!rectify) {
        cout << "Using pre-rectified images" << endl;
    }

    // Check we received at least two images
    if(images.size() < 2) {
        std::cerr << "Failed to capture first stereo pair from camera" << std::endl;
        return -1;
    }

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,K0(0,0),K0(1,1),K0(0,2),K0(1,2),0.1,10000),
        IdentityMatrix(GlModelViewStack)
    );

    GlBufferCudaPtr vbo(GlArrayBuffer, lw*lh,GL_FLOAT, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBufferCudaPtr cbo(GlArrayBuffer, lw*lh,GL_UNSIGNED_BYTE, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBuffer ibo = pangolin::MakeTriangleStripIboForVbo(lw,lh);

    // Allocate Camera Images on device for processing
    Image<unsigned char, TargetHost, DontManage> hCamImg[] = {{0,nw,nh},{0,nw,nh}};
    Image<float2, TargetDevice, Manage> dLookup[] = {{w,h},{w,h}};

    Image<unsigned char, TargetDevice, Manage> upload(w,h);
    Pyramid<unsigned char, max_levels, TargetDevice, Manage> img_pyr[] = {{w,h},{w,h}};

    Image<float, TargetDevice, Manage> img[] = {{lw,lh},{lw,lh}};
    Volume<float, TargetDevice, Manage> vol[] = {{lw,lh,MAXD},{lw,lh,MAXD}};
    Image<float, TargetDevice, Manage>  disp[] = {{lw,lh},{lw,lh}};
    Image<float, TargetDevice, Manage> meanI(lw,lh);
    Image<float, TargetDevice, Manage> varI(lw,lh);
    Image<float, TargetDevice, Manage> temp[] = {{lw,lh},{lw,lh},{lw,lh},{lw,lh},{lw,lh}};

    Image<float,TargetDevice, Manage>& imgd = disp[0];
    Image<float,TargetDevice, Manage> imga(lw,lh);
    Image<float2,TargetDevice, Manage> imgq(lw,lh);
    Image<float,TargetDevice, Manage> imgw(lw,lh);

    Image<float4, TargetDevice, Manage>  d3d(lw,lh);
    Image<unsigned char, TargetDevice,Manage> Scratch(lw*sizeof(LeastSquaresSystem<float,6>),lh);

    typedef ulong4 census_t;
    Image<census_t, TargetDevice, Manage> census[] = {{lw,lh},{lw,lh}};

    // Stereo transformation (post-rectification)
    Sophus::SE3 T_rl = T_rl_orig;

    const double baseline = T_rl.translation().norm();

    cudaMemGetInfo( &cu_mem_end, &cu_mem_total );
    cout << "CuTotal: " << cu_mem_total/(1024*1024) << ", Available: " << cu_mem_end/(1024*1024) << ", Used: " << (cu_mem_start-cu_mem_end)/(1024*1024) << endl;

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", false, true);
    Var<bool> lockToCam("ui.Lock to cam", false, true);
    Var<int> show_slice("ui.show slice",MAXD/2, 0, MAXD-1);

    Var<int> maxdisp("ui.maxdisp",MAXD, 0, MAXD);
    Var<bool> subpix("ui.subpix", true, true);

    Var<bool> use_census("ui.use census", true, true);
    Var<int> avg_rad("ui.avg_rad",0, 0, 100);

    Var<bool> do_dtam("ui.do dtam", false, true);
    Var<bool> dtam_reset("ui.reset", false, false);

    Var<float> g_alpha("ui.g alpha", 14, 0,4);
    Var<float> g_beta("ui.g beta", 2.5, 0,2);


    Var<float> theta("ui.theta", 100, 0,100);
    Var<float> lambda("ui.lambda", 20, 0,20);
    Var<float> sigma_q("ui.sigma q", 0.7, 0, 1);
    Var<float> sigma_d("ui.sigma d", 0.7, 0, 1);
    Var<float> huber_alpha("ui.huber alpha", 0.002, 0, 0.01);
    Var<float> beta("ui.beta", 0.00001, 0, 0.01);

    Var<float> alpha("ui.alpha", 0.9, 0,1);
    Var<float> r1("ui.r1", 100, 0,0.01);
    Var<float> r2("ui.r2", 100, 0,0.01);

    Var<bool> filter("ui.filter", false, true);
    Var<float> eps("ui.eps",0.01*0.01, 0, 0.01);
    Var<int> rad("ui.radius",9, 1, 20);

    Var<bool> leftrightcheck("ui.left-right check", false, true);
    Var<float> maxdispdiff("ui.maxdispdiff",1, 0, 5);

    Var<int> domedits("ui.median its",1, 1, 10);
    Var<bool> domed9x9("ui.median 9x9", false, true);
    Var<bool> domed7x7("ui.median 7x7", false, true);
    Var<bool> domed5x5("ui.median 5x5", false, true);
    Var<int> medi("ui.medi",12, 0, 24);

    Var<float> filtgradthresh("ui.filt grad thresh", 0, 0, 20);


    int jump_frames = 0;

    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
    pangolin::RegisterKeyPressCallback('l', [&lockToCam](){lockToCam = !lockToCam;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );
    pangolin::RegisterKeyPressCallback(']', [&jump_frames](){jump_frames=100;} );
    pangolin::RegisterKeyPressCallback('}', [&jump_frames](){jump_frames=1000;} );

    Handler2dImageSelect handler2d(lw,lh,level);
//    ActivateDrawPyramid<unsigned char,max_levels> adleft(img_pyr[0],GL_LUMINANCE8, false, true);
//    ActivateDrawPyramid<unsigned char,max_levels> adright(img_pyr[1],GL_LUMINANCE8, false, true);
    ActivateDrawImage<float> adleft(img[0],GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adright(img[1],GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adisp(disp[0],GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adw(imgw,GL_LUMINANCE32F_ARB, false, true);
//    ActivateDrawImage<float> adCrossSection(dCrossSection,GL_RGBA_FLOAT32_APPLE, false, true);
    ActivateDrawImage<float> adVol(vol[0].ImageXY(show_slice),GL_LUMINANCE32F_ARB, false, true);

    SceneGraph::GLSceneGraph graph;
    SceneGraph::GLVbo glvbo(&vbo,&ibo,&cbo);
    graph.AddChild(&glvbo);

    SetupContainer(container, 6, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adleft)).SetHandler(&handler2d);
    container[1].SetDrawFunction(boost::ref(adright)).SetHandler(&handler2d);
    container[2].SetDrawFunction(boost::ref(adisp)).SetHandler(&handler2d);
    container[3].SetDrawFunction(boost::ref(adVol)).SetHandler(&handler2d);
    container[4].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );
    container[5].SetDrawFunction(boost::ref(adw)).SetHandler(&handler2d);

    for(unsigned long frame=0; !pangolin::ShouldQuit();)
    {
        bool go = frame==0 || jump_frames > 0 || run || Pushed(step);

        for(; jump_frames > 0; jump_frames--) {
            video.Capture(images);
        }

        if(go) {
            if(frame>0) video.Capture(images);

            frame++;

            /////////////////////////////////////////////////////////////
            // Upload images to device (Warp / Decimate if necessery)
            for(int i=0; i<2; ++i ) {
                hCamImg[i].ptr = images[i].Image.data;

                if(rectify) {
                    upload.CopyFrom(hCamImg[i].SubImage(roi));
                    Warp(img_pyr[i][0], upload, dLookup[i]);
                }else{
                    img_pyr[i][0].CopyFrom(hCamImg[i].SubImage(roi));
                }

                BoxReduce<unsigned char, max_levels, unsigned int>(img_pyr[i]);
            }
        }

        go |= avg_rad.GuiChanged() | use_census.GuiChanged();
        if( go ) {
            for(int i=0; i<2; ++i ) {
                ElementwiseScaleBias<float,unsigned char,float>(img[i], img_pyr[i][level],1.0f/255.0f);
                if(avg_rad > 0 ) {
                    BoxFilter<float,float,float>(temp[0],img[i],Scratch,avg_rad);
                    ElementwiseAdd<float,float,float,float>(img[i], img[i], temp[0], 1, -1, 0.5);
                }
                if(use_census) {
                    Census(census[i], img[i]);
                }
            }
        }

        if( go | g_alpha.GuiChanged() || g_beta.GuiChanged() ) {
            ExponentialEdgeWeight(imgw, img[0], g_alpha, g_beta);
        }

        go |= filter.GuiChanged() | leftrightcheck.GuiChanged() | rad.GuiChanged() | eps.GuiChanged() | alpha.GuiChanged() | r1.GuiChanged() | r2.GuiChanged();
        if(go) {
            if(use_census) {
                CensusStereoVolume<float, census_t>(vol[0], census[0], census[1], maxdisp, -1);
                if(leftrightcheck) CensusStereoVolume<float, census_t>(vol[1], census[1], census[0], maxdisp, +1);
            }else{
                CostVolumeFromStereoTruncatedAbsAndGrad(vol[0], img[0], img[1], -1, alpha, r1, r2);
                if(leftrightcheck) CostVolumeFromStereoTruncatedAbsAndGrad(vol[1], img[1], img[0], +1, alpha, r1, r2);
            }

            if(filter) {
                // Filter Cost volume
                for(int v=0; v<(leftrightcheck?2:1); ++v)
                {
                    Image<float, TargetDevice, Manage>& I = img[v];
                    ComputeMeanVarience<float,float,float>(varI, temp[0], meanI, I, Scratch, rad);

                    for(int d=0; d<maxdisp; ++d)
                    {
                        Image<float> P = vol[v].ImageXY(d);
                        ComputeCovariance(temp[0],temp[2],temp[1],P,meanI,I,Scratch,rad);
                        GuidedFilter(P,temp[0],varI,temp[1],meanI,I,Scratch,temp[2],temp[3],temp[4],rad,eps);
                    }
                }
            }
        }

        static int n = 0;
//        static float theta = 0;
//        go |= Pushed(dtam_reset);
//        if(go )
        if(Pushed(dtam_reset))
        {
            n = 0;
            theta.Reset();

            // Initialise primal and auxillary variables
            CostVolMinimumSubpix(imgd,vol[0], maxdisp,-1);
            imga.CopyFrom(imgd);

            // Initialise dual variable
            imgq.Memset(0);
        }

        if(do_dtam && theta > 1E-3)
        {
            for(int i=0; i<5; ++i ) {
                // Auxillary exhaustive search
                CostVolMinimumSquarePenaltySubpix(imga, vol[0], imgd, maxdisp, -1, lambda, (theta) );

                // Dual Ascent
                Gpu::WeightedHuberGradU_DualAscentP(imgq, imgd, imgw, sigma_q, huber_alpha);

                // Primal Descent
                Gpu::WeightedL2_u_minus_g_PrimalDescent(imgd, imgq, imga, imgw, sigma_d, 1.0f / (theta) );

                theta= theta * (1-beta*n);
                ++n;
            }
        }

        go |= GuiVarHasChanged();
//        if(go) {
//            if(subpix) {
//                CostVolMinimumSubpix(disp[0],vol[0], maxdisp,-1);
//                if(leftrightcheck) CostVolMinimumSubpix(disp[1],vol[1], maxdisp,+1);
//            }else{
//                CostVolMinimum<float,float>(disp[0],vol[0], maxdisp);
//                if(leftrightcheck) CostVolMinimum<float,float>(disp[1],vol[1], maxdisp);
//            }

//        }

        if(go) {
            for(int di=0; di<(leftrightcheck?2:1); ++di) {
                for(int i=0; i < domedits; ++i ) {
                    if(domed9x9) MedianFilterRejectNegative9x9(disp[di],disp[di], medi);
                    if(domed7x7) MedianFilterRejectNegative7x7(disp[di],disp[di], medi);
                    if(domed5x5) MedianFilterRejectNegative5x5(disp[di],disp[di], medi);
                }
            }

            if(leftrightcheck ) {
                LeftRightCheck(disp[1], disp[0], +1, maxdispdiff);
                LeftRightCheck(disp[0], disp[1], -1, maxdispdiff);
            }

            if(filtgradthresh > 0) {
                FilterDispGrad(disp[0], disp[0], filtgradthresh);
            }
        }

//        if(go)
        {
            // Generate point cloud from disparity image
            DisparityImageToVbo(d3d, disp[0], baseline, Kl(0,0), Kl(1,1), Kl(0,2), Kl(1,2) );

//            if(container[3].IsShown())
            {
                // Copy point cloud into VBO
                {
                    CudaScopedMappedPtr var(vbo);
                    Gpu::Image<float4> dVbo((float4*)*var,lw,lh);
                    dVbo.CopyFrom(d3d);
                }

                // Generate CBO
                {
                    CudaScopedMappedPtr var(cbo);
                    Gpu::Image<uchar4> dCbo((uchar4*)*var,lw,lh);
                    ConvertImage<uchar4,unsigned char>(dCbo, img_pyr[0][level]);
                }
            }

            // Update texture views
            adisp.SetImageScale(1.0f/maxdisp);
//            adleft.SetLevel(show_level);
//            adright.SetLevel(show_level);
            adVol.SetImage(vol[0].ImageXY(show_slice));
        }

        /////////////////////////////////////////////////////////////
        // Draw

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

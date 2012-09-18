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
#include "common/BaseDisplayCuda.h"
#include "common/HeightmapFusion.h"
#include "common/CameraModelPyramid.h"
#include "common/LoadPosesFromFile.h"
#include "common/SavePPM.h"

#include <kangaroo/kangaroo.h>

#include <Node.h>

//#define HM_FUSION
//#define PLANE_FIT
//#define COSTVOL_TIME

const int MAXD = 128;

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
    CameraDevice video = OpenRpgCamera(argc,argv);

    // Capture first image
    std::vector<rpg::ImageWrapper> images;
    video.Capture(images);

    // native width and height (from camera)
    const unsigned int nw = images[0].width();
    const unsigned int nh = images[0].height();

    // Downsample this image to process less pixels
    const int max_levels = 6;
    const int level = GetLevelFromMaxPixels( nw, nh, 2*640*480 );
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

    // Load history
    Sophus::SE3 T_wc;
    vector<Sophus::SE3> gtPoseT_wh;
    LoadPosesFromFile(gtPoseT_wh, video.GetProperty("DataSourceDir") + "/pose.txt", video.GetProperty("StartFrame",0), T_vis_ro, T_ro_vis);

#ifdef PLANE_FIT
    // Plane Parameters
    // These coordinates need to be below the horizon. This could cause trouble!
    Eigen::Matrix3d U; U << w, 0, w,  h/2, h, h,  1, 1, 1;
    Eigen::Matrix3d Q = -(cam[0].Kinv() * U).transpose();
    Eigen::Matrix3d Qinv = Q.inverse();
    Eigen::Vector3d z; z << 1/5.0, 1/5.0, 1/5.0;
    Eigen::Vector3d n_c = Qinv*z;
    Eigen::Vector3d n_w = project((Eigen::Vector4d)(T_wc.inverse().matrix().transpose() * unproject(n_c)));
#endif // PLANE_FIT

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,K0(0,0),K0(1,1),K0(0,2),K0(1,2),0.1,10000),
        IdentityMatrix(GlModelViewStack)
    );
    if(!gtPoseT_wh.empty()) {
        s_cam.SetModelViewMatrix(gtPoseT_wh[0].inverse().matrix());
    }

    GlBufferCudaPtr vbo(GlArrayBuffer, lw,lh,GL_FLOAT, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBufferCudaPtr cbo(GlArrayBuffer, lw,lh,GL_UNSIGNED_BYTE, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBufferCudaPtr ibo(GlElementArrayBuffer, lw,lh,GL_UNSIGNED_INT, 2 );

    // Generate Index Buffer Object for rendering mesh
    {
        CudaScopedMappedPtr var(ibo);
        Gpu::Image<uint2> dIbo((uint2*)*var,lw,lh);
        GenerateTriangleStripIndexBuffer(dIbo);
    }


    // Allocate Camera Images on device for processing
    Image<unsigned char, TargetHost, DontManage> hCamImg[] = {{0,nw,nh},{0,nw,nh}};
    Image<float2, TargetDevice, Manage> dLookup[] = {{w,h},{w,h}};

    Image<unsigned char, TargetDevice, Manage> upload(w,h);
    Pyramid<unsigned char, max_levels, TargetDevice, Manage> img_pyr[] = {{w,h},{w,h}};

    Image<float, TargetDevice, Manage> img[] = {{lw,lh},{lw,lh}};
    Volume<float, TargetDevice, Manage> vol[] = {{lw,lh,MAXD},{lw,lh,MAXD},{lw,lh,MAXD}};
    Image<float, TargetDevice, Manage>  disp[] = {{lw,lh},{lw,lh}};
    Image<float, TargetDevice, Manage> meanI(lw,lh);
    Image<float, TargetDevice, Manage> varI(lw,lh);
    Image<float, TargetDevice, Manage> temp[] = {{lw,lh},{lw,lh},{lw,lh},{lw,lh},{lw,lh}};

    Image<float4, TargetDevice, Manage>  d3d(lw,lh);
    Image<float, TargetDevice, Manage>  dCrossSection(lw,MAXD);
    Image<unsigned char, TargetDevice,Manage> Scratch(lw*sizeof(LeastSquaresSystem<float,6>),lh);
    Image<float, TargetDevice, Manage>  Err(lw,lh);

    typedef ulong4 census_t;
    Image<census_t, TargetDevice, Manage> census[] = {{lw,lh},{lw,lh}};

    Image<unsigned char, TargetHost, Manage> hImg[] = {{lw,lh},{lw,lh}};
    Image<float, TargetHost, Manage> hDisp(lw,lh);

#ifdef COSTVOL_TIME
    Sophus::SE3 T_wv;
    Volume<CostVolElem, TargetDevice, Manage>  dCostVol(lw,lh,MAXD);
    Image<unsigned char, TargetDevice, Manage> dImgv(lw,lh);
#endif

#ifdef HM_FUSION
//    HeightmapFusion hm(800,800,2);
//    HeightmapFusion hm(200,200,10);
    HeightmapFusion hm(100,100,10);
    const bool center_y = false;

    GlBufferCudaPtr vbo_hm(GlArrayBuffer, hm.WidthPixels(), hm.HeightPixels(), GL_FLOAT, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBufferCudaPtr cbo_hm(GlArrayBuffer, hm.WidthPixels(), hm.HeightPixels(), GL_UNSIGNED_BYTE, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
    GlBufferCudaPtr ibo_hm(GlElementArrayBuffer, hm.WidthPixels(), hm.HeightPixels(), GL_UNSIGNED_INT, 2 );

    //generate index buffer for heightmap
    {
        CudaScopedMappedPtr var(ibo_hm);
        Gpu::Image<uint2> dIbo((uint2*)*var,hm.WidthPixels(),hm.HeightPixels());
        GenerateTriangleStripIndexBuffer(dIbo);
    }
#endif // HM_FUSION

    // Stereo transformation (post-rectification)
    Sophus::SE3 T_rl = T_rl_orig;

    // Build camera distortion lookup tables
    if(rectify) {
        T_rl = CreateScanlineRectifiedLookupAndT_rl(
                    dLookup[0], dLookup[1], T_rl_orig,
                    cam[0].K(), cam[0].GetModel()->warped.kappa1, cam[0].GetModel()->warped.kappa2,
                    cam[1].K(), cam[1].GetModel()->warped.kappa1, cam[1].GetModel()->warped.kappa2
                    );
    }

    const double baseline = T_rl.translation().norm();

    cudaMemGetInfo( &cu_mem_end, &cu_mem_total );
    cout << "CuTotal: " << cu_mem_total/(1024*1024) << ", Available: " << cu_mem_end/(1024*1024) << ", Used: " << (cu_mem_start-cu_mem_end)/(1024*1024) << endl;

    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", false, true);
    Var<bool> save_dm("ui.save_dm", false);
    Var<bool> record("ui.record_dm", false,false);
    Var<bool> lockToCam("ui.Lock to cam", false, true);
    Var<int> show_level("ui.show level",0, 0, max_levels-1);
    Var<int> show_slice("ui.show slice",MAXD/2, 0, MAXD-1);

    Var<int> maxdisp("ui.maxdisp",MAXD, 0, MAXD);
    Var<bool> subpix("ui.subpix", true, true);

    Var<bool> use_census("ui.use census", false, true);

    Var<float> alpha("ui.alpha", 0.9, 0,1);
    Var<float> r1("ui.r1", 0.0028, 0,0.01);
    Var<float> r2("ui.r2", 0.008, 0,0.01);

    Var<bool> filter("ui.filter", true, true);
    Var<float> eps("ui.eps",0.01*0.01, 0, 0.01);
    Var<int> rad("ui.radius",9, 1, 20);

    Var<bool> applyBilateralFilter("ui.Apply Bilateral Filter", false, true);
    Var<int> bilateralWinSize("ui.size",18, 1, 20);
    Var<float> gs("ui.gs",10, 1E-3, 10);
    Var<float> gr("ui.gr",6, 1E-3, 10);
    Var<float> gc("ui.gc",0.01, 1E-4, 0.1);

    Var<bool> do_sgm_h("ui.SGM horiz", false, true);
    Var<bool> do_sgm_v("ui.SGM vert", false, true);
    Var<bool> do_sgm_reverse("ui.SGM reverse", false, true);
    Var<float> sgm_p1("ui.sgm p1",0.01, 0, 0.1);
    Var<float> sgm_p2("ui.sgm p2",0.02, 0, 1, false);

    Var<bool> leftrightcheck("ui.left-right check", true, true);
    Var<float> maxdispdiff("ui.maxdispdiff",1, 0, 5);

    Var<int> domedits("ui.median its",1, 1, 10);
    Var<bool> domed9x9("ui.median 9x9", false, true);
    Var<bool> domed7x7("ui.median 7x7", false, true);
    Var<bool> domed5x5("ui.median 5x5", false, true);
    Var<int> medi("ui.medi",12, 0, 24);

    Var<float> filtgradthresh("ui.filt grad thresh", 0, 0, 20);

    Var<int> avg_rad("ui.avg_rad",5, 1, 100);


#ifdef PLANE_FIT
    Var<bool> resetPlane("ui.resetplane", true, false);
    Var<bool> plane_do("ui.Compute Ground Plane", false, true);
    Var<float> plane_within("ui.Plane Within",20, 0.1, 100);
    Var<float> plane_c("ui.Plane c", 0.5, 0.0001, 1);
#endif // PLANE_FIT

#ifdef HM_FUSION
    Var<bool> fuse("ui.fuse", false, true);
    Var<bool> save_hm("ui.save heightmap", false, false);
#endif // HM_FUSION

//    Var<bool> draw_frustrum("ui.show frustrum", false, true);
//    Var<bool> show_mesh("ui.show mesh", true, true);
//    Var<bool> show_color("ui.show color", true, true);
    Var<bool> show_history("ui.show history", true, true);
    Var<bool> show_depthmap("ui.show depthmap", true, true);

#ifdef HM_FUSION
    Var<bool> show_heightmap("ui.show heightmap", false, true);
#endif // HM_FUSION

#ifdef COSTVOL_TIME
    Var<bool> cross_section("ui.Cross Section", true, true);
    Var<bool> costvol_reset("ui.Costvol Reset", true, false);
    Var<bool> costvol_reset_stereo("ui.Costvol Reset Stereo", false, false);
    Var<bool> costvol_add("ui.Add to Costvol", false, false);
#endif

    int jump_frames = 0;

    pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
    pangolin::RegisterKeyPressCallback('l', [&lockToCam](){lockToCam = !lockToCam;} );
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + GLUT_KEY_RIGHT, [&step](){step=true;} );
    pangolin::RegisterKeyPressCallback(']', [&jump_frames](){jump_frames=100;} );
    pangolin::RegisterKeyPressCallback('}', [&jump_frames](){jump_frames=1000;} );
    pangolin::RegisterKeyPressCallback('~', [&container](){static bool showpanel=true; showpanel = !showpanel; if(showpanel) { container.SetBounds(0,1,Attach::Pix(180), 1); }else{ container.SetBounds(0,1,0, 1); } Display("ui").Show(showpanel); } );
    pangolin::RegisterKeyPressCallback('1', [&container](){container[0].ToggleShow();} );
    pangolin::RegisterKeyPressCallback('2', [&container](){container[1].ToggleShow();} );
    pangolin::RegisterKeyPressCallback('3', [&container](){container[2].ToggleShow();} );
    pangolin::RegisterKeyPressCallback('4', [&container](){container[3].ToggleShow();} );
    pangolin::RegisterKeyPressCallback('$', [&container](){container[3].SaveRenderNow("screenshot",4);} );

    Handler2dImageSelect handler2d(lw,lh,level);
//    ActivateDrawPyramid<unsigned char,max_levels> adleft(img_pyr[0],GL_LUMINANCE8, false, true);
//    ActivateDrawPyramid<unsigned char,max_levels> adright(img_pyr[1],GL_LUMINANCE8, false, true);
    ActivateDrawImage<float> adleft(img[0],GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adright(img[1],GL_LUMINANCE32F_ARB, false, true);
    ActivateDrawImage<float> adisp(disp[0],GL_LUMINANCE32F_ARB, false, true);
//    ActivateDrawImage<float> adCrossSection(dCrossSection,GL_RGBA_FLOAT32_APPLE, false, true);
    ActivateDrawImage<float> adVol(vol[0].ImageXY(show_slice),GL_LUMINANCE32F_ARB, false, true);

    SceneGraph::GLSceneGraph graph;
    SceneGraph::GLVbo glvbo(&vbo,&ibo,&cbo);

    SceneGraph::GLCameraHistory history;
    history.LoadFromAbsoluteCartesianFile(video.GetProperty("DataSourceDir") + "/pose.txt", video.GetProperty("StartFrame",0), T_vis_ro, T_ro_vis);
    graph.AddChild(&glvbo);
    graph.AddChild(&history);

#ifdef PLANE_FIT
    SceneGraph::GLGrid glGroundPlane;
    glvbo.AddChild(&glGroundPlane);
#endif

#ifdef HM_FUSION
    SceneGraph::GLVbo glhmvbo(&vbo_hm,&ibo_hm,&cbo_hm);
    graph.AddChild(&glhmvbo);
#endif

    SetupContainer(container, 5, (float)w/h);
    container[0].SetDrawFunction(boost::ref(adleft)).SetHandler(&handler2d);
    container[1].SetDrawFunction(boost::ref(adright)).SetHandler(&handler2d);
    container[2].SetDrawFunction(boost::ref(adisp)).SetHandler(&handler2d);
    container[3].SetDrawFunction(boost::ref(adVol)).SetHandler(&handler2d);
    container[4].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, s_cam))
                .SetHandler( new Handler3D(s_cam, AxisNone) );

    for(unsigned long frame=0; !pangolin::ShouldQuit();)
    {
        const bool go = frame==0 || jump_frames > 0 || run || Pushed(step);

        for(; jump_frames > 0; jump_frames--) {
            video.Capture(images);
        }

        if(go) {
            if(frame>0) video.Capture(images);

            if(frame < gtPoseT_wh.size()) {
                T_wc = gtPoseT_wh[frame];
            }

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
            }
        }

        if(go || GuiVarHasChanged() )
        {
            for(int i=0; i<2; ++i ) {
                BoxReduce<unsigned char, max_levels, unsigned int>(img_pyr[i]);
                ElementwiseScaleBias<float,unsigned char,float>(img[i], img_pyr[i][level],1.0f/255.0f);
//                BoxFilter<float,float,float>(temp[0],img[i],Scratch,avg_rad);
//                ElementwiseAdd<float,float,float,float>(img[i], img[i], temp[0], 1, -1, 0.5);

                Census(census[i], img[i]);
            }

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

            if(applyBilateralFilter) {
                // Filter Cost volume
                for(int v=0; v<(leftrightcheck?2:1); ++v)
                {
                    Image<float, TargetDevice, Manage>& I = img[v];

                    for(int d=0; d<maxdisp; ++d)
                    {
                        Image<float> P = vol[v].ImageXY(d);
                        temp[0].CopyFrom(P);
                        BilateralFilter<float,float,float>(P,temp[0],I,gs,gr,gc,bilateralWinSize);
                    }
                }
            }

            if(do_sgm_h || do_sgm_v) {
                for(int i=0; i<1; ++i) {
                    SemiGlobalMatching<float,float,float>(vol[2],vol[i],img[i], maxdisp, sgm_p1, sgm_p2, do_sgm_h, do_sgm_v, do_sgm_reverse);
                    vol[i].CopyFrom(vol[2]);
                }
            }

            if(subpix) {
                CostVolMinimumSubpix(disp[0],vol[0], maxdisp,-1);
                if(leftrightcheck) CostVolMinimumSubpix(disp[1],vol[1], maxdisp,+1);
            }else{
                CostVolMinimum<float,float>(disp[0],vol[0], maxdisp);
                if(leftrightcheck) CostVolMinimum<float,float>(disp[1],vol[1], maxdisp);
            }

            for(int di=0; di<(leftrightcheck?2:1); ++di) {
                for(int i=0; i < domedits; ++i ) {
                    if(domed9x9) MedianFilterRejectNegative9x9(disp[di],disp[di], medi);
                    if(domed7x7) MedianFilterRejectNegative7x7(disp[di],disp[di], medi);
                    if(domed5x5) MedianFilterRejectNegative5x5(disp[di],disp[di], medi);
                }
            }

//            if(applyBilateralFilter) {
//                temp[0].CopyFrom(disp[0]);
//                BilateralFilter<float,float,float>(disp[0],temp[0],img[0],gs,gr,gc,bilateralWinSize);
//            }

            if(leftrightcheck ) {
                LeftRightCheck(disp[1], disp[0], +1, maxdispdiff);
                LeftRightCheck(disp[0], disp[1], -1, maxdispdiff);
            }

            if(filtgradthresh > 0) {
                FilterDispGrad(disp[0], disp[0], filtgradthresh);
            }

#ifdef COSTVOL_TIME
            if(Pushed(costvol_reset)) {
                T_wv = T_wc;
                dImgv.CopyFrom(img_pyr[0][level]);
                CostVolumeZero(dCostVol);
            }

            if(Pushed(costvol_reset_stereo)) {
                T_wv = T_wc;
                dImgv.CopyFrom(img_pyr[0][level]);
                CostVolumeFromStereo(dCostVol,img_pyr[0][level], img_pyr[1][level]);
            }

            if(Pushed(costvol_add)) {
                const Eigen::Matrix<double,3,4> KT_lv = Kl * (T_wc.inverse() * T_wv).matrix3x4();
                CostVolumeAdd(dCostVol,dImgv, img_pyr[0][level], KT_lv, Kl(0,0), Kl(1,1), Kl(0,2), Kl(1,2), baseline, 0);
            }

            // Extract Minima of cost volume
//            CostVolMinimum(disp[0], dCostVol);
#endif // COSTVOL_TIME

            if(go && save_dm) {
                // file info
                char Index[10];
                sprintf( Index, "%05d", frame );
                string sFileName = (std::string)"disp/depth-" + Index + ".pdm";

                // Save Disparity Image
                hDisp.CopyFrom(disp[0]);
                SavePXM<float>(sFileName, hDisp, "P7", maxdisp);

                if(rectify)
                {
                    // Save rectified images
                    hImg[0].CopyFrom(img_pyr[0][0]);
                    SavePXM<unsigned char>((std::string)"disp/left-" + Index + ".pgm", hImg[0], "P5");
                    hImg[1].CopyFrom(img_pyr[1][0]);
                    SavePXM<unsigned char>((std::string)"disp/right-" + Index + ".pgm", hImg[1], "P5");
                }
            }

            // Generate point cloud from disparity image
            DisparityImageToVbo(d3d, disp[0], baseline, Kl(0,0), Kl(1,1), Kl(0,2), Kl(1,2) );

#ifdef PLANE_FIT
            if(plane_do || resetPlane) {
                // Fit plane
                for(int i=0; i<(resetPlane*100+5); ++i )
                {
                    Gpu::LeastSquaresSystem<float,3> lss = PlaneFitGN(d3d, Qinv, z, Scratch, Err, plane_within, plane_c);
                    Eigen::FullPivLU<Eigen::Matrix3d> lu_JTJ( (Eigen::Matrix3d)lss.JTJ );
                    Eigen::Vector3d x = -1.0 * lu_JTJ.solve( (Eigen::Vector3d)lss.JTy );
                    if( x.norm() > 1 ) x = x / x.norm();
                    for(int i=0; i<3; ++i ) {
                        z(i) *= exp(x(i));
                    }
                    n_c = Qinv * z;
                    n_w = project((Eigen::Vector4d)(T_wc.inverse().matrix().transpose() * unproject(n_c)));
                }
            }
#endif // PLANE_FIT

#ifdef HM_FUSION
            if(Pushed(resetPlane) ) {
                Eigen::Matrix4d T_nw = (PlaneBasis_wp(n_c).inverse() * T_wc.inverse()).matrix();
//                T_nw.block<2,1>(0,3) += Eigen::Vector2d(hm.WidthMeters()/2, hm.HeightMeters() /*/2*/);
                T_nw.block<2,1>(0,3) += Eigen::Vector2d(hm.WidthMeters()/2, hm.HeightMeters() / (center_y ? 2 : 1) );
                hm.Init(T_nw);
            }

            //calcualte the camera to heightmap transform
            if(fuse) {
                hm.Fuse(d3d, img_pyr[0][level], T_wc);
                hm.GenerateVbo(vbo_hm);
                hm.GenerateCbo(cbo_hm);
            }

            if(Pushed(save_hm)) {
                hm.SaveModel("test");
            }
#elif PLANE_FIT
            resetPlane = false;
#endif

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
#ifdef COSTVOL_TIME
                    ConvertImage<uchar4,unsigned char>(dCbo, dImgv);
#else
                    ConvertImage<uchar4,unsigned char>(dCbo, img_pyr[0][level]);
#endif
                }
            }

            // Update texture views
            adisp.SetImageScale(1.0f/maxdisp);
//            adleft.SetLevel(show_level);
//            adright.SetLevel(show_level);
            adVol.SetImage(vol[0].ImageXY(show_slice));
        }

#ifdef COSTVOL_TIME
        if(cross_section) {
            CostVolumeCrossSection(dCrossSection, dCostVol, handler2d.GetSelectedPoint(true)[1] + 0.5);
        }
#endif // COSTVOL_TIME

        if(Pushed(record)) {
            // Start recording next frame
            frame = 0;
            video.InitDriver("FileReader");
            video.Capture(images);
            save_dm = true;
        }

        /////////////////////////////////////////////////////////////
        // Setup Drawing

        s_cam.Follow(T_wc.matrix(), lockToCam);

#ifdef COSTVOL_TIME
        glvbo.SetPose(T_wv.matrix());
        glvbo.SetVisible(show_depthmap);
#else
        glvbo.SetPose(T_wc.matrix());
#endif


#ifdef PLANE_FIT
        glGroundPlane.SetPose(PlaneBasis_wp(n_c).matrix());
        glGroundPlane.SetVisible(plane_do);
#endif // PLANE_FIT

#ifdef HM_FUSION
        glhmvbo.SetPose((Eigen::Matrix4d)hm.T_hw().inverse());
        glhmvbo.SetVisible(show_heightmap);
#endif // HM_FUSION

        history.SetNumberToShow(frame);
        history.SetVisible(show_history);

        /////////////////////////////////////////////////////////////
        // Draw

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        pangolin::FinishGlutFrame();
    }
}

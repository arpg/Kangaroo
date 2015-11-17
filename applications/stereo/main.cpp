#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

#include <pangolin/pangolin.h>
#include <pangolin/compat/function.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glsl.h>

#include <SceneGraph/SceneGraph.h>
#include <SceneGraph/GLVbo.h>

#include <calibu/Calibu.h>

#include <kangaroo/extra/RpgCameraOpen.h>
#include <kangaroo/extra/ImageSelect.h>
#include <kangaroo/extra/BaseDisplayCuda.h>
#include <kangaroo/extra/BaselineFromCamModel.h>

#include <kangaroo/kangaroo.h>
#include <kangaroo/variational.h>

const int MAXD = 64;


int main( int argc, char* argv[] )
{
  // Initialise window
  pangolin::View& container = SetupPangoGLWithCuda(1024, 768);
  size_t cu_mem_start, cu_mem_end, cu_mem_total;
  cudaMemGetInfo( &cu_mem_start, &cu_mem_total );
  glClearColor(1,1,1,0);

  // Open video device
  hal::Camera video = OpenRpgCamera(argc,argv);

  // Capture first image
  std::shared_ptr<hal::ImageArray> images = hal::ImageArray::Create();

  // N cameras, each w*h in dimension, greyscale
  const size_t N = video.NumChannels();
  if( N < 2 ) {
    std::cerr << "Two images are required to run this program!" << std::endl;
    exit(1);
  }
  const size_t nw = video.Width();
  const size_t nh = video.Height();

  // Capture first image
  video.Capture(*images);

  // Downsample this image to process less pixels
  const int max_levels = 6;
  const int level = roo::GetLevelFromMaxPixels( nw, nh, 640*480 );
  //    const int level = 4;
  assert(level <= max_levels);

  // Find centered image crop which aligns to 16 pixels at given level
  const NppiRect roi = roo::GetCenteredAlignedRegion(nw,nh,16 << level,16 << level);

  // Load Camera intrinsics from file
  GetPot clArgs( argc, argv );
  bool export_time = clArgs.search("-export_time");
  const std::string filename = clArgs.follow("","-cmod");
  if( filename.empty() ) {
    std::cerr << "Camera models file is required!" << std::endl;
    exit(1);
  }
  std::shared_ptr<calibu::Rig<double>> rig = calibu::ReadXmlRig(filename);

  if( rig->NumCams() != 2 ) {
    std::cerr << "Two camera models are required to run this program!" << std::endl;
    exit(1);
  }

  Eigen::Matrix3f CamModel0 = rig->cameras_[0]->K().cast<float>();
  Eigen::Matrix3f CamModel1 = rig->cameras_[1]->K().cast<float>();

  roo::ImageIntrinsics camMod[] = {
    {CamModel0(0,0),CamModel0(1,1),CamModel0(0,2),CamModel0(1,2)},
    {CamModel1(0,0),CamModel1(1,1),CamModel1(0,2),CamModel1(1,2)}
  };

  for(int i=0; i<2; ++i ) {
    // Adjust to match camera image dimensions
    const double scale = nw / rig->cameras_[i]->Width();
    roo::ImageIntrinsics camModel = camMod[i].Scale( scale );

    // Adjust to match cropped aligned image
    camModel = camModel.CropToROI( roi );

    camMod[i] = camModel;
  }

  const unsigned int w = roi.width;
  const unsigned int h = roi.height;
  const unsigned int lw = w >> level;
  const unsigned int lh = h >> level;

  const Eigen::Matrix3d& K0 = camMod[0].Matrix();
  const Eigen::Matrix3d& Kl = camMod[0][level].Matrix();

  std::cout << "K Matrix: " << std::endl << K0 << std::endl;
  std::cout << "K Matrix - Level: " << std::endl << Kl << std::endl;

  std::cout << "Video stream dimensions: " << nw << "x" << nh << std::endl;
  std::cout << "Chosen Level: " << level << std::endl;
  std::cout << "Processing dimensions: " << lw << "x" << lh << std::endl;
  std::cout << "Offset: " << roi.x << "x" << roi.y << std::endl;

  // print selected camera model
  std::cout << "Camera Model used: " << std::endl << camMod[0][level].Matrix() << std::endl;

  Eigen::Matrix3d RDFvision;RDFvision<< 1,0,0,  0,1,0,  0,0,1;
  Eigen::Matrix3d RDFrobot; RDFrobot << 0,1,0,  0,0, 1,  1,0,0;
  Eigen::Matrix4d T_vis_ro = Eigen::Matrix4d::Identity();
  T_vis_ro.block<3,3>(0,0) = RDFvision.transpose() * RDFrobot;
  Eigen::Matrix4d T_ro_vis = Eigen::Matrix4d::Identity();
  T_ro_vis.block<3,3>(0,0) = RDFrobot.transpose() * RDFvision;

  const Sophus::SE3d T_rl_orig = T_rlFromCamModelRDF(rig->cameras_[0],
                                  rig->cameras_[1], RDFvision);

  // TODO(jmf): For now, assume cameras are rectified. Later allow unrectified cameras.
  /*
    double k1 = 0;
    double k2 = 0;

    if(cam[0].Type() == MVL_CAMERA_WARPED)
    {
        k1 = cam[0].GetModel()->warped.kappa1;
        k2 = cam[0].GetModel()->warped.kappa2;
    }
    */

  const bool rectify = false;
  if(!rectify) {
    std::cout << "Using pre-rectified images" << std::endl;
  }

  // Check we received at least two images
  if(images->Size() < 2) {
    std::cerr << "Failed to capture first stereo pair from camera" << std::endl;
    return -1;
  }

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrixRDF_TopLeft(w,h,K0(0,0),K0(1,1),K0(0,2),K0(1,2),0.1,10000),
        pangolin::IdentityMatrix(pangolin::GlModelViewStack)
        );

  pangolin::GlBufferCudaPtr vbo(pangolin::GlArrayBuffer, lw*lh,GL_FLOAT, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
  pangolin::GlBufferCudaPtr cbo(pangolin::GlArrayBuffer, lw*lh,GL_UNSIGNED_BYTE, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW );
  pangolin::GlBuffer ibo = pangolin::MakeTriangleStripIboForVbo(lw,lh);

  // Allocate Camera Images on device for processing
  roo::Image<unsigned char, roo::TargetHost, roo::DontManage> hCamImg[] = {{0,nw,nh},{0,nw,nh}};
  roo::Image<float2, roo::TargetDevice, roo::Manage> dLookup[] = {{w,h},{w,h}};

  roo::Image<unsigned char, roo::TargetDevice, roo::Manage> upload(w,h);
  roo::Pyramid<unsigned char, max_levels, roo::TargetDevice, roo::Manage> img_pyr[] = {{w,h},{w,h}};

  roo::Image<float, roo::TargetDevice, roo::Manage> img[] = {{lw,lh},{lw,lh}};
  roo::Volume<float, roo::TargetDevice, roo::Manage> vol[] = {{lw,lh,MAXD},{lw,lh,MAXD}};
  roo::Image<float, roo::TargetDevice, roo::Manage>  disp[] = {{lw,lh},{lw,lh}};
  roo::Image<float, roo::TargetDevice, roo::Manage> meanI(lw,lh);
  roo::Image<float, roo::TargetDevice, roo::Manage> varI(lw,lh);
  roo::Image<float, roo::TargetDevice, roo::Manage> temp[] = {{lw,lh},{lw,lh},{lw,lh},{lw,lh},{lw,lh}};

  roo::Image<float,roo::TargetDevice, roo::Manage>& imgd = disp[0];
  roo::Image<float,roo::TargetDevice, roo::Manage> depthmap(lw,lh);
  roo::Image<float,roo::TargetDevice, roo::Manage> imga(lw,lh);
  roo::Image<float2,roo::TargetDevice, roo::Manage> imgq(lw,lh);
  roo::Image<float,roo::TargetDevice, roo::Manage> imgw(lw,lh);

  roo::Image<float4, roo::TargetDevice, roo::Manage>  d3d(lw,lh);
  roo::Image<unsigned char, roo::TargetDevice,roo::Manage> Scratch(lw*sizeof(roo::LeastSquaresSystem<float,6>),lh);

  typedef ulong4 census_t;
  roo::Image<census_t, roo::TargetDevice, roo::Manage> census[] = {{lw,lh},{lw,lh}};

  // Stereo transformation (post-rectification)
  Sophus::SE3d T_rl = T_rl_orig;

  const double baseline = T_rl.translation().norm();
  std::cout << "Baseline: " << baseline << std::endl;

  cudaMemGetInfo( &cu_mem_end, &cu_mem_total );
  std::cout << "CuTotal: " << cu_mem_total/(1024*1024) << ", Available: " << cu_mem_end/(1024*1024) << ", Used: " << (cu_mem_start-cu_mem_end)/(1024*1024) << std::endl;

  pangolin::Var<bool> step("ui.step", false, false);
  pangolin::Var<bool> run("ui.run", false, true);
  pangolin::Var<bool> lockToCam("ui.Lock to cam", false, true);
  pangolin::Var<int> show_slice("ui.show slice",MAXD/2, 0, MAXD-1);

  pangolin::Var<int> maxdisp("ui.maxdisp",MAXD, 0, MAXD);
  pangolin::Var<bool> subpix("ui.subpix", true, true);

  pangolin::Var<bool> use_census("ui.use census", true, true);
  pangolin::Var<int> avg_rad("ui.avg_rad",0, 0, 100);

  pangolin::Var<bool> do_dtam("ui.do dtam", false, true);
  pangolin::Var<bool> dtam_reset("ui.reset", false, false);

  pangolin::Var<float> g_alpha("ui.g alpha", 14, 0,4);
  pangolin::Var<float> g_beta("ui.g beta", 2.5, 0,2);


  pangolin::Var<float> theta("ui.theta", 100, 0,100);
  pangolin::Var<float> lambda("ui.lambda", 20, 0,20);
  pangolin::Var<float> sigma_q("ui.sigma q", 0.7, 0, 1);
  pangolin::Var<float> sigma_d("ui.sigma d", 0.7, 0, 1);
  pangolin::Var<float> huber_alpha("ui.huber alpha", 0.002, 0, 0.01);
  pangolin::Var<float> beta("ui.beta", 0.00001, 0, 0.01);

  pangolin::Var<float> alpha("ui.alpha", 0.9, 0,1);
  pangolin::Var<float> r1("ui.r1", 100, 0,0.01);
  pangolin::Var<float> r2("ui.r2", 100, 0,0.01);

  pangolin::Var<bool> filter("ui.filter", false, true);
  pangolin::Var<float> eps("ui.eps",0.01*0.01, 0, 0.01);
  pangolin::Var<int> rad("ui.radius",9, 1, 20);

  pangolin::Var<bool> leftrightcheck("ui.left-right check", false, true);
  pangolin::Var<float> maxdispdiff("ui.maxdispdiff",1, 0, 5);

  pangolin::Var<int> domedits("ui.median its",1, 1, 10);
  pangolin::Var<bool> domed9x9("ui.median 9x9", false, true);
  pangolin::Var<bool> domed7x7("ui.median 7x7", false, true);
  pangolin::Var<bool> domed5x5("ui.median 5x5", false, true);
  pangolin::Var<int> medi("ui.medi",12, 0, 24);

  pangolin::Var<float> filtgradthresh("ui.filt grad thresh", 0, 0, 20);

  pangolin::Var<bool> save_depthmaps("ui.save_depthmaps", false, true);

  int jump_frames = 0;

  pangolin::RegisterKeyPressCallback(' ', [&run](){run = !run;} );
  pangolin::RegisterKeyPressCallback('l', [&lockToCam](){lockToCam = !lockToCam;} );
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT, [&step](){step=true;} );
  pangolin::RegisterKeyPressCallback(']', [&jump_frames](){jump_frames=100;} );
  pangolin::RegisterKeyPressCallback('}', [&jump_frames](){jump_frames=1000;} );

  pangolin::Handler2dImageSelect handler2d(lw,lh,level);
  //    ActivateDrawPyramid<unsigned char,max_levels> adleft(img_pyr[0],GL_LUMINANCE8, false, true);
  //    ActivateDrawPyramid<unsigned char,max_levels> adright(img_pyr[1],GL_LUMINANCE8, false, true);
  pangolin::ActivateDrawImage<float> adleft(img[0],GL_LUMINANCE32F_ARB, false, true);
  pangolin::ActivateDrawImage<float> adright(img[1],GL_LUMINANCE32F_ARB, false, true);
  pangolin::ActivateDrawImage<float> adisp(disp[0],GL_LUMINANCE32F_ARB, false, true);
  pangolin::ActivateDrawImage<float> adw(imgw,GL_LUMINANCE32F_ARB, false, true);
  //    ActivateDrawImage<float> adCrossSection(dCrossSection,GL_RGBA_FLOAT32_APPLE, false, true);
  pangolin::ActivateDrawImage<float> adVol(vol[0].ImageXY(show_slice),GL_LUMINANCE32F_ARB, false, true);

  SceneGraph::GLSceneGraph graph;
  //    SceneGraph::GLVbo glvbo(&vbo,&ibo,&cbo);
  SceneGraph::GLVbo glvbo(&vbo,nullptr,&cbo);
  graph.AddChild(&glvbo);

  SetupContainer(container, 6, (float)w/h);
  container[0].SetDrawFunction(boostd::ref(adleft)).SetHandler(&handler2d);
  container[1].SetDrawFunction(boostd::ref(adright)).SetHandler(&handler2d);
  container[2].SetDrawFunction(boostd::ref(adisp)).SetHandler(&handler2d);
  container[3].SetDrawFunction(boostd::ref(adVol)).SetHandler(&handler2d);
  container[4].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, s_cam))
      .SetHandler( new pangolin::Handler3D(s_cam, pangolin::AxisNone) );
  container[5].SetDrawFunction(boostd::ref(adw)).SetHandler(&handler2d);

  bool bFirstTime = true;

  for(unsigned long frame=0; !pangolin::ShouldQuit();)
  {
    bool go = frame==0 || jump_frames > 0 || run || Pushed(step);

    for(; jump_frames > 0; jump_frames--) {
      video.Capture(*images);
    }

    if(go) {
      if(frame>0) {
        if( video.Capture(*images) == false) {
          exit(1);
        }
      }

      frame++;

      /////////////////////////////////////////////////////////////
      // Upload images to device (Warp / Decimate if necessery)
      for(int i=0; i<2; ++i ) {
        hCamImg[i].ptr = (unsigned char*) images->at(i)->data();

        if(rectify) {
          upload.CopyFrom(hCamImg[i].SubImage(roi));
          Warp(img_pyr[i][0], upload, dLookup[i]);
        }else{
          img_pyr[i][0].CopyFrom(hCamImg[i].SubImage(roi));
        }

        roo::BoxReduce<unsigned char, max_levels, unsigned int>(img_pyr[i]);
      }
    }

    go |= avg_rad.GuiChanged() | use_census.GuiChanged();
    if( go ) {
      for(int i=0; i<2; ++i ) {
        roo::ElementwiseScaleBias<float,unsigned char,float>(img[i], img_pyr[i][level],1.0f/255.0f);
        if(avg_rad > 0 ) {
          roo::BoxFilter<float,float,float>(temp[0],img[i],Scratch,avg_rad);
          roo::ElementwiseAdd<float,float,float,float>(img[i], img[i], temp[0], 1, -1, 0.5);
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
        roo::CensusStereoVolume<float, census_t>(vol[0], census[0], census[1], maxdisp, -1);
        if(leftrightcheck) roo::CensusStereoVolume<float, census_t>(vol[1], census[1], census[0], maxdisp, +1);
      }else{
        CostVolumeFromStereoTruncatedAbsAndGrad(vol[0], img[0], img[1], -1, alpha, r1, r2);
        if(leftrightcheck) CostVolumeFromStereoTruncatedAbsAndGrad(vol[1], img[1], img[0], +1, alpha, r1, r2);
      }

      if(filter) {
        // Filter Cost volume
        for(int v=0; v<(leftrightcheck?2:1); ++v)
        {
          roo::Image<float, roo::TargetDevice, roo::Manage>& I = img[v];
          roo::ComputeMeanVarience<float,float,float>(varI, temp[0], meanI, I, Scratch, rad);

          for(int d=0; d<maxdisp; ++d)
          {
            roo::Image<float> P = vol[v].ImageXY(d);
            roo::ComputeCovariance(temp[0],temp[2],temp[1],P,meanI,I,Scratch,rad);
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

    const double min_theta = 1E-4;
    if(do_dtam && theta > min_theta)
    {
      for(int i=0; i<5; ++i ) {

        // Dual Ascent
        roo::WeightedHuberGradU_DualAscentP(imgq, imgd, imgw, sigma_q, huber_alpha);

        // Primal Descent
        roo::WeightedL2_u_minus_g_PrimalDescent(imgd, imgq, imga, imgw, sigma_d, 1.0f / (theta) );

        // Auxillary exhaustive search
        CostVolMinimumSquarePenaltySubpix(imga, vol[0], imgd, maxdisp, -1, lambda, (theta) );

        theta= theta * (1-beta*n);
        ++n;
      }
      if( theta <= min_theta && save_depthmaps ) {
        cv::Mat dmap = cv::Mat( lh, lw, CV_32FC1 );
        // convert disparity to depth
        roo::Disp2Depth(imgd, depthmap, Kl(0,0), baseline );
        depthmap.MemcpyToHost( dmap.data );

        // save depth image
        const double    timestamp = images->SystemTime();
        int             frame_i = static_cast<int>(frame);
        char            index[30];

        if (export_time) {
          sprintf(index, "%015.10f", timestamp);
        } else {
          sprintf(index, "%05d", frame_i);
        }
        std::string DepthPrefix = "SDepth-";
        std::string DepthFile;
        DepthFile = DepthPrefix + index + ".pdm";
        std::cout << "Depth File: " << DepthFile << std::endl;
        std::ofstream pDFile( DepthFile.c_str(), std::ios::out | std::ios::binary );
        pDFile << "P7" << std::endl;
        pDFile << dmap.cols << " " << dmap.rows << std::endl;
        unsigned int Size = dmap.elemSize1() * dmap.rows * dmap.cols;
        pDFile << 4294967295 << std::endl;
        pDFile.write( (const char*)dmap.data, Size );
        pDFile.close();

        // save grey image
        std::string GreyPrefix = "Left-";
        std::string GreyFile;
        GreyFile = GreyPrefix + index + ".pgm";
        std::cout << "Grey File: " << GreyFile << std::endl;
        cv::Mat gimg = cv::Mat( lh, lw, CV_8UC1 );
        img_pyr[0][level].MemcpyToHost( gimg.data );
        cv::imwrite( GreyFile, gimg );

        // reset
        step = true;
        dtam_reset = true;
      }

      if (bFirstTime) {
        float* temp;
        std::cout<<"n = "<<n<<std::endl;
        temp = (float *) malloc(2*sizeof(float)*imgq.h*imgq.w);
        imgq.MemcpyToHost(temp);
        for(int jj = 0; jj < 5; jj++) {
          for(int ii = 0; ii < 5; ii++){
            int index = 2*(ii + jj*imgq.w);
            std::cout<<"("<< temp[index] <<","<< temp[index + 1] <<")\t";
          }
          std::cout<<std::endl;
        }
        free(temp);
        bFirstTime = false;
      }

    }


    go |= pangolin::GuiVarHasChanged();
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
          pangolin::CudaScopedMappedPtr var(vbo);
          roo::Image<float4> dVbo((float4*)*var,lw,lh);
          dVbo.CopyFrom(d3d);
        }

        // Generate CBO
        {
          pangolin::CudaScopedMappedPtr var(cbo);
          roo::Image<uchar4> dCbo((uchar4*)*var,lw,lh);
          roo::ConvertImage<uchar4,unsigned char>(dCbo, img_pyr[0][level]);
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

    pangolin::FinishFrame();
  }
}

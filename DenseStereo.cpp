#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <sophus/se3.h>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include "RpgCameraOpen.h"
#include "kernel.h"

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>

using namespace std;
using namespace pangolin;
using namespace Gpu;

template<typename T, typename Owner>
inline void operator<<(pangolin::GlTextureCudaArray& tex, const Image<T,TargetDevice,Owner>& dImg)
{
    CudaScopedMappedArray arr_tex(tex);
    cudaMemcpy2DToArray(*arr_tex, 0, 0, dImg.ptr, dImg.pitch, dImg.w*sizeof(T), dImg.h, cudaMemcpyDeviceToDevice );
}

Eigen::Matrix3d MakeK(const Eigen::VectorXd& camParamsVec, size_t w, size_t h)
{
    Eigen::Matrix3d K;
    K << camParamsVec(0)*w, 0, camParamsVec(2)*w,
            0, camParamsVec(1)*h, camParamsVec(3)*h,
            0,0,1;
    return K;
}

Eigen::Matrix3d MakeKinv(const Eigen::Matrix3d& K)
{
    Eigen::Matrix3d Kinv = Eigen::Matrix3d::Identity();
    Kinv << 1.0/K(0,0), 0, - K(0,2) / K(0,0),
            0, 1.0/K(1,1), - K(1,2) / K(1,1),
            0,0,1;
    return Kinv;
}

Sophus::SE3 CreateScanlineRectifiedLookupAndT_rl(
    Image<float2> dlookup_left, Image<float2> dlookup_right,
    const Sophus::SE3 T_rl, const Eigen::Matrix3d& K, double kappa1, double kappa2,
    size_t w, size_t h
) {
//    Eigen::Matrix3d K =    MakeK(camParamsVec, w, h);
    Eigen::Matrix3d Kinv = MakeKinv(K);

    const Sophus::SO3 R_rl = T_rl.so3();
    const Sophus::SO3 R_lr = R_rl.inverse();
    const Eigen::Vector3d l_r = T_rl.translation();
    const Eigen::Vector3d r_l = - (R_lr * l_r);

    // Current up vector for each camera (in left FoR)
    const Eigen::Vector3d lup_l = Eigen::Vector3d(0,1,0);
    const Eigen::Vector3d rup_l = R_lr * Eigen::Vector3d(0,1,0);

    // Hypothetical fwd vector for each camera, perpendicular to baseline (in left FoR)
    const Eigen::Vector3d lfwd = lup_l.cross(r_l);
    const Eigen::Vector3d rfwd = rup_l.cross(r_l);

    // New fwd is average of left / right hypothetical baselines (also perpendicular to baseline)
    const Eigen::Vector3d new_fwd = (lfwd + rfwd).normalized();

    // Define new basis (in left FoR);
    const Eigen::Vector3d x = r_l.normalized();
    const Eigen::Vector3d z = -new_fwd;
    const Eigen::Vector3d y  = z.cross(x).normalized();

    // New orientation for both left and right cameras (expressed relative to original left)
    Eigen::Matrix3d mR_nl;
    mR_nl << x, y, z;

    // By definition, the right camera now lies exactly on the x-axis with the same orientation
    // as the left camera.
    const Sophus::SE3 T_nr_nl = Sophus::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(-r_l.norm(),0,0) );


    // Homographies which should be applied to left and right images to scan-line rectify them
    const Eigen::Matrix3d Hl_nl = K * mR_nl.transpose() * Kinv;
    const Eigen::Matrix3d Hr_nr = K * (mR_nl * R_lr.matrix()).transpose() * Kinv;

    // Copy to simple Array objects to pass to CUDA by Value
    Gpu::Array<float,9> H_ol_nl;
    Gpu::Array<float,9> H_or_nr;

    for(int r=0; r<3; ++r) {
        for(int c=0; c<3; ++c) {
            H_ol_nl[3*r+c] = Hl_nl(r,c);
            H_or_nr[3*r+c] = Hr_nr(r,c);
        }
    }

    // Invoke CUDA Kernel to generate lookup table
    CreateMatlabLookupTable(dlookup_left, K(0,0), K(1,1), K(0,2), K(1,2), kappa1, kappa2, H_ol_nl);
    CreateMatlabLookupTable(dlookup_right,K(0,0), K(1,1), K(0,2), K(1,2), kappa1, kappa2, H_or_nr);

    return T_nr_nl;
}

Sophus::SE3 T_rlFromCamModelRDF(const mvl::CameraModel& lcmod, const mvl::CameraModel& rcmod, const Eigen::Matrix3d& targetRDF)
{
    // Transformation matrix to adjust to target RDF
    Eigen::Matrix4d Tadj[2] = {Eigen::Matrix4d::Identity(),Eigen::Matrix4d::Identity()};
    Tadj[0].block<3,3>(0,0) = targetRDF.transpose() * lcmod.RDF();
    Tadj[1].block<3,3>(0,0) = targetRDF.transpose() * rcmod.RDF();

    // Computer Poses in our adjust coordinate system
    const Eigen::Matrix4d T_lw_ = Tadj[0] * lcmod.GetPose().inverse();
    const Eigen::Matrix4d T_rw_ = Tadj[1] * rcmod.GetPose().inverse();

    // Computer transformation to right camera frame from left
    const Eigen::Matrix4d T_rl = T_rw_ * T_lw_.inverse();

    return Sophus::SE3(T_rl.block<3,3>(0,0), T_rl.block<3,1>(0,3) );
}

int main( int /*argc*/, char* argv[] )
{
    // Open video device
//    const std::string cam_uri =
    CameraDevice camera = OpenRpgCamera(
//        "AlliedVision:[NumChannels=2,CamUUID0=5004955,CamUUID1=5004954,ImageBinningX=2,ImageBinningY=2,ImageWidth=694,ImageHeight=518]//"
        "FileReader:[NumChannels=2,DataSourceDir=/Users/slovegrove/data/CityBlock-Noisy,Channel-0=left.*pgm,Channel-1=right.*pgm,StartFrame=0]//"
//        "FileReader:[NumChannels=2,DataSourceDir=/Users/slovegrove/data/xb3,Channel-0=left.*pgm,Channel-1=right.*pgm,StartFrame=0]//"
//        "Dvi2Pci:[NumImages=2,ImageWidth=640,ImageHeight=480,BufferCount=60]//"
    );

//    CameraDevice camera = OpenPangoCamera(
//        "file:[stream=0,fmt=GRAY8]///Users/slovegrove/data/3DCam/DSCF0051.AVI",
//        "file:[stream=1,fmt=GRAY8]///Users/slovegrove/data/3DCam/DSCF0051.AVI"
//    );

    mvl::CameraModel camModel[] = {
        camera.GetProperty("DataSourceDir") + "/lcmod.xml",
        camera.GetProperty("DataSourceDir") + "/rcmod.xml"
    };

    Eigen::Matrix3d RDFgl;
    RDFgl << 1,0,0,  0,-1,0,  0,0,-1;

    const Sophus::SE3 T_rl_orig = T_rlFromCamModelRDF(camModel[0], camModel[1], RDFgl);
    double k1 = 0;
    double k2 = 0;

    if(camModel[0].Type() == MVL_CAMERA_WARPED)
    {
        k1 = camModel[0].GetModel()->warped.kappa1;
        k2 = camModel[0].GetModel()->warped.kappa2;
    }

    const bool rectify = (k1!=0 || k2!=0); // || camModel[0].GetPose().block<3,3>(0,0)

    // Capture first image
    std::vector<rpg::ImageWrapper> img;
    camera.Capture(img);

    // Check we received one or more images
    if(img.empty()) {
        std::cerr << "Failed to capture first image from camera" << std::endl;
        return -1;
    }

    // N cameras, each w*h in dimension, greyscale
    const unsigned int w = img[0].width();
    const unsigned int h = img[0].height();

    // Setup OpenGL Display (based on GLUT)
    pangolin::CreateGlutWindowAndBind(__FILE__,2*w,2*h);

    // Initialise CUDA, allowing it to use OpenGL context
    cudaGLSetGLDevice(0);

    // Setup default OpenGL parameters
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_BLEND);
    glEnable (GL_LINE_SMOOTH);
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    // Tell the base view to arrange its children equally
    const int UI_WIDTH = 180;
    View& d_panel = pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, Attach::Pix(UI_WIDTH));

    View& container = CreateDisplay()
            .SetBounds(0,1.0, Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(LayoutEqual);

    const int N = 4;
    for(int i=0; i<N; ++i ) {
        View& disp = CreateDisplay().SetAspect((double)w/h);
        container.AddDisplay(disp);
    }

    // Texture we will use to display camera images
    GlTextureCudaArray tex8(w,h,GL_LUMINANCE8);
    GlTextureCudaArray texrgb8(w,h,GL_RGBA8);
    GlTextureCudaArray texf(w,h,GL_LUMINANCE32F_ARB);

    // Allocate Camera Images on device for processing
    Image<unsigned char, TargetDevice, Manage> dCamImgDist[] = {{w,h},{w,h}};
    Image<unsigned char, TargetDevice, Manage> dCamImg[] = {{w,h},{w,h}};
    Image<float2, TargetDevice, Manage> dLookup[] = {{w,h},{w,h}};
    Image<uchar4, TargetDevice, Manage> d3d(w,h);
    Image<float, TargetDevice, Manage>  dDisp(640,480);
    Image<float, TargetDevice, Manage>  dDispFilt(640,480);

    // Camera Parameters
//    Eigen::VectorXd camParamsVec(6);
//    camParamsVec << 0.558526, 0.747774, 0.484397, 0.494393, -0.249261, 0.0825967;

    // Stereo transformation (post-rectification)
    Sophus::SE3 T_rl = T_rl_orig;

    // Build camera distortion lookup tables
    if(rectify)
    {
//        // Actual Original Stereo configuration
//        Eigen::Matrix3d mR_rl_orig;
//        mR_rl_orig << 0.999995,   0.00188482,  -0.00251896,
//                -0.0018812,     0.999997,   0.00144025,
//                0.00252166,  -0.00143551,     0.999996;

//        Eigen::Vector3d l_r_orig;
//        l_r_orig <<    -0.203528, -0.000750334, 0.00403201;

//        const Sophus::SO3 R_rl_orig = Sophus::SO3(mR_rl_orig);
//        const Sophus::SE3 T_rl_orig = Sophus::SE3(R_rl_orig, l_r_orig);

        T_rl = CreateScanlineRectifiedLookupAndT_rl(
                    dLookup[0], dLookup[1], T_rl_orig,
                    camModel[0].K(), k1, k2, w, h
                    );
    }

    static Var<int> disp("ui.disp",0, 0, 64);
    static Var<int> size("ui.size",5, 1, 20);
    static Var<float> gs("ui.gs",1E-3, 1E-3, 5);
    static Var<float> gr("ui.gr",1E-3, 1E-3, 5);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor3f(1,1,1);

        camera.Capture(img);

        /////////////////////////////////////////////////////////////
        // Upload images to device
        for(int i=0; i<2; ++i ) {
            if(rectify) {
                dCamImgDist[i].MemcpyFromHost(img[i].Image.data, w*sizeof(uchar1) );
                Warp(dCamImg[i],dCamImgDist[i],dLookup[i]);
            }else{
                dCamImg[i].MemcpyFromHost(img[i].Image.data, w*sizeof(uchar1) );
            }
        }

        DenseStereo(dDisp, dCamImg[0], dCamImg[1], disp);
        BilateralFilter(dDispFilt, dDisp, gs, gr, size);
//        MakeAnaglyth(d3d, dCamImg[0], dCamImg[1]);

        /////////////////////////////////////////////////////////////
        // Perform drawing
        // Draw Stereo images
        for(int i=0; i<2; ++i ) {
            container[i].Activate();
            tex8 << dCamImg[i];
            tex8.RenderToViewportFlipY();
        }

        container[2].Activate();
        texf << dDisp;
        texf.RenderToViewportFlipY();

        container[3].Activate();
        texf << dDispFilt;
        texf.RenderToViewportFlipY();

        d_panel.Render();

        pangolin::FinishGlutFrame();
    }
}

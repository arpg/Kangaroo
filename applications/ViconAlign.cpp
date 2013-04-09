#include <pangolin/pangolin.h>
#include <pangolin/video.h>
#include <pangolin/video_record_repeat.h>
#include <pangolin/input_record_repeat.h>

#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/OpenGLSupport>

#ifdef HAVE_FPL
#include <CameraModel.h>
#include <CCameraModel/GridCalibrator.h>
using namespace CCameraModel;
#endif

#include "common/ViconTracker.h"
#include "common/RpgCameraOpen.h"
#include "common/BaseDisplay.h"

#include <Mvlpp/Mvl.h>
#include <Mvlpp/Cameras.h>


using namespace std;
using namespace pangolin;
using namespace Eigen;

//#include <boost/thread.hpp>

inline double Tic()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * (tv.tv_usec);
}


struct Observation
{
    Eigen::MatrixXd obs;
    Sophus::SE3d T_fw;
};

Eigen::Matrix<double,2,3> dpi_dx(const Eigen::Vector3d& x)
{
    const double x2x2 = x(2)*x(2);
    Eigen::Matrix<double,2,3> ret;
    ret << 1.0 / x(2), 0,  -x(0) / x2x2,
            0, 1.0 / x(2), -x(1) / x2x2;
    return ret;
}

Eigen::Matrix<double,4,4> se3_gen(unsigned i) {

    Eigen::Matrix<double,4,4> ret;
    ret.setZero();

    switch(i) {
    case 0: ret(0,3) = 1; break;
    case 1: ret(1,3) = 1; break;
    case 2: ret(2,3) = 1; break;
    case 3: ret(1,2) = -1; ret(2,1) = 1; break;
    case 4: ret(0,2) = 1; ret(2,0) = -1; break;
    case 5: ret(0,1) = -1; ret(1,0) = 1; break;
    }

    return ret;
}

double err(
        const MatlabCamera& cam,
        const RandomDotTarget& target,
        const std::vector<Observation>& vicon_obs,
        const Sophus::SE3d& T_cf,
        const Sophus::SE3d& T_wt
) {
    int num_seen = 0;
    double sumsqerr = 0;

    for( int i=0; i< vicon_obs.size(); ++i ) {
        const Observation& sample = vicon_obs[i];
        for( int j=0; j < target.circles3D().size(); ++j ) {
            Eigen::Vector3d P_t = target.circles3D()[j];
            Eigen::Vector3d p_c_ = cam.K() * (T_cf * sample.T_fw * T_wt * P_t);
            Eigen::Vector2d p_c = project( (Eigen::Vector3d)(p_c_ ) );
            Eigen::Vector2d obs = project( (Eigen::Vector3d)(cam.K() * cam.unmap_unproject(sample.obs.col(j)) ) );
            Eigen::Vector2d err = p_c - obs;
            double sqerr = err.squaredNorm();

            if(isfinite(sqerr)) {
                num_seen++;
                sumsqerr += sqerr;
            }
        }
    }

    return sumsqerr / num_seen;
}

void OptimiseTargetVicon(
    const MatlabCamera& cam,
    const RandomDotTarget& target,
    const std::vector<Observation>& vicon_obs,
    Sophus::SE3d& T_cf,
    Sophus::SE3d& T_wt
) {
    int num_seen;
    double sumsqerr;
    Eigen::Matrix<double,12,12> JTJ;
    Eigen::Matrix<double,12,1> JTy;

//    for( int it = 0; it < 1; ++it )
    {
        num_seen = 0;
        sumsqerr = 0;
        JTJ.setZero();
        JTy.setZero();

        for( int i=0; i< vicon_obs.size(); ++i ) {
            const Observation& sample = vicon_obs[i];
            for( int j=0; j < target.circles3D().size(); ++j ) {
                Eigen::Vector3d P_t = target.circles3D()[j];
                Eigen::Vector3d p_c_ = cam.K() * (T_cf * sample.T_fw * T_wt * P_t);
                Eigen::Vector2d p_c = project( (Eigen::Vector3d)(p_c_ ) );
                Eigen::Vector2d obs = project( (Eigen::Vector3d)(cam.K() * cam.unmap_unproject(sample.obs.col(j)) ) );
                Eigen::Vector2d err = p_c - obs;
                double sqerr = err.squaredNorm();

                if(isfinite(sqerr)) {
                    num_seen++;
                    sumsqerr += sqerr;

                    const Eigen::Matrix<double,2,3> dpi = dpi_dx(p_c_);
                    const Eigen::Matrix<double,2,4> mi1 = dpi * cam.K() * T_cf.matrix().block<3,4>(0,0);
                    const Eigen::Matrix<double,4,1> mi2 = unproject( (Vector3d)(sample.T_fw * T_wt * P_t) );
                    const Eigen::Matrix<double,2,4> mj1 = mi1 * (sample.T_fw * T_wt).matrix();
                    const Eigen::Matrix<double,4,1> mj2 = unproject( (Vector3d)(P_t) );

                    Eigen::Matrix<double,12,2> J;
                    for(int i=0; i<6; ++i) {
                        J.row(i) = mi1 * se3_gen(i) * mi2;
                        J.row(i+6) = mj1 * se3_gen(i) * mj2;
                    }

                    JTJ += J.col(0) * J.col(0).transpose();
                    JTJ += J.col(1) * J.col(1).transpose();
                    JTy += J.col(0) * err(0);
                    JTy += J.col(1) * err(1);
                }
            }
        }

        cout << sumsqerr / num_seen << endl;

        JTJ.ldlt().solveInPlace(JTy);

        T_cf = T_cf * Sophus::SE3d::exp(-1.0 * JTy.segment<6>(0));
        T_wt = T_wt * Sophus::SE3d::exp(-1.0 * JTy.segment<6>(6));
    }
}

namespace Sophus {
template<typename T, int R, int C>
std::istream& operator>> (std::istream& is, Eigen::Matrix<T,R,C>& o){
    for(int r=0; r < R; ++r )  {
        for(int c=0; c < C; ++c )  {
            is >> o(r,c);
        }
    }
    return is;
}

template<typename T>
std::ostream& operator<< (std::ostream& os, const Eigen::Quaternion<T>& o){
    os << o.w() << " " << o.x() << " " << o.y() << " " << o.z() << endl;
    return os;
}
template<typename T>
std::istream& operator>> (std::istream& is, Eigen::Quaternion<T>& o){
    is >> o.w();
    is >> o.x();
    is >> o.y();
    is >> o.z();
    return is;
}

std::ostream& operator<< (std::ostream& os, const Sophus::SE3d& o){
    const Eigen::Vector3d& t = o.translation();
    os << t(0) << " " << t(1) << " " << t(2) << " " << o.unit_quaternion();
    return os;
}

std::istream& operator>> (std::istream& is, Sophus::SE3d& o){
    Eigen::Quaterniond q;
    is >> o.translation()(0);
    is >> o.translation()(1);
    is >> o.translation()(2);
    is >> q;
    o.setQuaternion(q);
    return is;
}

}


int main( int argc, char* argv[] )
{
    ViconTracking vicon("GENLEE","192.168.10.1");
    std::vector<Observation> vicon_obs;

    // vicon frame to camera transform
    Sophus::SE3d T_cf;

    // Target to World transform
    Sophus::SE3d T_wt;

    CameraDevice video;
    OpenRpgCamera(video,argc,argv);
    std::vector<rpg::ImageWrapper> images;
    video.Capture(images);
    const unsigned w = images[0].width();
    const unsigned h = images[0].height();

    // Setup Tracker and associated target
    TrackerParams tracker_params;
    Tracker tracker(w,h);

    // Create Target in Meters
//    const Eigen::Vector2d targetSizeMeters = Eigen::Vector2d(11,8.5) * 0.0254;
//    const double trad = targetSizeMeters[0]/40;
//    tracker.target.GenerateRandom(60, trad, 3*trad, trad, targetSizeMeters);
//    tracker.target.SaveRotatedEPS("target.eps",72/0.0254);

    tracker.target.LoadEPS("/home/rpg/Code/Builds/Kangaroo/applications/stereo.eps", 72/0.0254);

    // Pangolin 3D Render state
    pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(640,480,420,420,320,240,1E-3,1E6),
        ModelViewLookAtRDF(0,5,5,0,0,0,0,0,1)
    );
    pangolin::Handler3D handler(s_cam);

    // Create Glut window
    View& container = SetupPangoGL(1024, 768);
    SetupContainer(container,3,w/(float)h);

    // Create viewport for video with fixed aspect
    View& vVideo = container[0];
    View& v3D = container[1];
    View& v3D2 = container[2];
    v3D.SetHandler(&handler);
    v3D2.SetHandler(&handler);

    // OpenGl Texture for video frame
    GlTexture tex(w,h,GL_LUMINANCE8);

    // Camera parameters
//    Matrix<double,9,1> cam_params; // = Var<Matrix<double,9,1> >("cam_params");
//    cam_params << 0.808936, 1.06675, 0.495884, 0.520504, 0.180668, -0.354284, -0.00169838, 0.000600873, 0.0;
    //  FovCamera cam( w,h, w*cam_params[0],h*cam_params[1], w*cam_params[2],h*cam_params[3], cam_params[4] );
    //MatlabCamera cam( w,h, w*cam_params[0],h*cam_params[1], w*cam_params[2],h*cam_params[3], cam_params[4], cam_params[5], cam_params[6], cam_params[7], cam_params[8]);
    mvl::CameraModel cammodel("/home/rpg/Data/Flea3/lcmod.xml");
    MatlabCamera cam( w,h, cammodel.GetModel()->warped.fx, cammodel.GetModel()->warped.fy, cammodel.GetModel()->warped.cx, cammodel.GetModel()->warped.cy, cammodel.GetModel()->warped.kappa1, cammodel.GetModel()->warped.kappa2, cammodel.GetModel()->warped.tau1, cammodel.GetModel()->warped.tau2, cammodel.GetModel()->warped.kappa3 );

    // Variables
    Var<bool> step("ui.step", false, false);
    Var<bool> run("ui.run", true, true);

    Var<bool> disp_thresh("ui.Display Thresh",false);
    Var<bool> lock_to_cam("ui.AR",false);
    Var<bool> add_image("ui.add Image",false,false);
    Var<bool> guess("ui.guess calib",false,false);
    Var<bool> minimise_vicon("ui.minimise vicon",false,false);
    Var<bool> reset("ui.reset",false,false);

#ifdef HAVE_FPL
    Var<bool> minimise("ui.minimise calib",false,false);
    Eigen::MatrixXd pattern = tracker.TargetPattern3D();
    GridCalibrator calibrator("PinholeRadTan", w, h, pattern);
#endif

    Var<Sophus::SE3d> vicon_T_wf("vicon.T_wf");

    double rms = 0;
    Var<double> var_rms("ui.rms");

    pangolin::RegisterKeyPressCallback(' ',  [&add_image](){add_image = true;} );
    pangolin::RegisterKeyPressCallback('\\', [&minimise_vicon](){minimise_vicon = true;} );
    pangolin::RegisterKeyPressCallback('r',  [&reset](){reset = true;} );

    for(int frame=0; !pangolin::ShouldQuit();)
    {
        const bool go = frame==0 || run || Pushed(step);

        var_rms = rms;

        Viewport::DisableScissor();
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        if(go) {
            if(frame>0) video.Capture(images);
            frame++;
        }

        vicon_T_wf = vicon.T_wf();

        const bool tracking_good =
                tracker.ProcessFrame(tracker_params,cam,images[0].Image.data);

        if(Pushed(guess) || (add_image && vicon_obs.size()==0) ) {
            Eigen::Matrix3d RDFvision;RDFvision<< 1,0,0,  0,1,0,   0,0,1;
            Eigen::Matrix3d RDFvicon; RDFvicon << 0,1,0,  0,0,1,   1,0,0;
            T_cf = Sophus::SE3d(Sophus::SO3d(RDFvision.transpose() * RDFvicon), Eigen::Vector3d::Zero() );
//            T_cf = Sophus::SE3d();
            T_wt = (Sophus::SE3d)vicon_T_wf * T_cf.inverse() * tracker.T_gw;
            cout << err(cam, tracker.target, vicon_obs, T_cf, T_wt) << endl;
        }

        if( Pushed(add_image) ) {
            if( tracker.NumVisibleFeatures() > tracker.target.NumCircles() - 10 )
            {
              const Eigen::MatrixXd obs = tracker.TargetPatternObservations();

              // Generate visible list
              std::vector<short int> visibleCircles;
              for(int i=0; i< obs.cols(); ++i ) {
                  if( isfinite(obs(0,0)) ) {
                      visibleCircles.push_back(i);
                  }
              }

#ifdef HAVE_FPL
              calibrator.add_view(obs, visibleCircles);
#endif

              vicon_obs.push_back((Observation){obs,((Sophus::SE3d)vicon_T_wf).inverse() });
              cout << err(cam, tracker.target, vicon_obs, T_cf, T_wt) << endl;
            }
        }

#ifdef HAVE_FPL
        if(Pushed(minimise)) {
            calibrator.minimise();
            calibrator.save("camparams.txt");

            cout <<
                    calibrator.get_camera_copy()->get<double>("fx") / w << " " <<
                    calibrator.get_camera_copy()->get<double>("fy") / h << " " <<
                    calibrator.get_camera_copy()->get<double>("cx") / w << " " <<
                    calibrator.get_camera_copy()->get<double>("cy") / h << " " <<
                    calibrator.get_camera_copy()->get<double>("k1") << " " <<
                    calibrator.get_camera_copy()->get<double>("k2") << " " <<
                    calibrator.get_camera_copy()->get<double>("p1") << " " <<
                    calibrator.get_camera_copy()->get<double>("p2") << " 0.0" << endl;
        }
#endif

        if(Pushed(minimise_vicon)) {
            OptimiseTargetVicon(cam,tracker.target,vicon_obs, T_cf, T_wt);
            cout << "Vicon to camera (T_cv)" << endl;
//            const Eigen::Quaterniond q = T_cf.so3().unit_quaternion();
//            const Eigen::Vector3d t = T_cf.translation();
//            cout << "Sophus::SE3d(Sophus::SO3d(Eigen::Quaterniond(" << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ")), Eigen::Vector3d(" << t[0] << "," << t[1] << "," << t[2] << ") )" << endl;
            cout << "T_cv" << mvl::T2Cart(T_cf.matrix()).transpose() << endl;
        }

        if(Pushed(reset)) {
            vicon_obs.clear();
            T_cf = Sophus::SE3d();
            T_wt = Sophus::SE3d();
        }

        //    calibrator.iterate(rms);
        s_cam.Follow(tracker.T_gw.matrix(),lock_to_cam);

        // Display Live Image
        glColor3f(1,1,1);

        if(vVideo.IsShown()) {
            vVideo.ActivateScissorAndClear();

            if(!disp_thresh) {
                tex.Upload(images[0].Image.data,GL_LUMINANCE,GL_UNSIGNED_BYTE);
                tex.RenderToViewportFlipY();
            }else{
                tex.Upload(tracker.tI.get(),GL_LUMINANCE,GL_UNSIGNED_BYTE);
                tex.RenderToViewportFlipY();
            }

            // Display detected ellipses
            glOrtho(-0.5,w-0.5,h-0.5,-0.5,0,1.0);
            for( int i=0; i<tracker.conics.size(); ++i ) {
                glColorBin(tracker.conics_target_map[i],tracker.target.circles3D().size());
                DrawCross(tracker.conics[i].center,2);
            }
        }

        // Display 3D Vis
        if(v3D.IsShown()) {
            glEnable(GL_DEPTH_TEST);
            v3D.ActivateScissorAndClear(s_cam);
            glDepthFunc(GL_LEQUAL);
            glDrawAxis(0.1);
            DrawTarget(tracker.target,Vector2d(0,0),1,0.2,0.2);
            DrawTarget(tracker.conics_target_map,tracker.target,Vector2d(0,0),1);

            //    if( tracking_good )
            {
                // Draw Camera
                glColor3f(1,0,0);
                DrawFrustrum(cam.Kinv(),w,h,tracker.T_hw.inverse(),0.1);
            }
        }

        if(v3D2.IsShown()) {
            v3D2.ActivateScissorAndClear(s_cam);

            glColor3f(0.5,0.5,0.5);
            DrawGrid(20,0.25);

            glDisable(GL_DEPTH_TEST);
            glColor3f(0.8,0.8,0.8);
            DrawGrid(5,1.0);
            glDrawAxis(1);
            glEnable(GL_DEPTH_TEST);

            // Draw Vicon
            glSetFrameOfReferenceF(vicon_T_wf);
            glDrawAxis(0.1);
            DrawFrustrum(cam.Kinv(),w,h,T_cf.inverse(),0.1);
            glUnsetFrameOfReference();

            // Draw Target
            glSetFrameOfReferenceF(T_wt);

            {
                DrawTarget(tracker.target,Vector2d(0,0),1,0.2,0.2);
                DrawTarget(tracker.conics_target_map,tracker.target,Vector2d(0,0),1);
                glColor3f(1,0,0);
                DrawFrustrum(cam.Kinv(),w,h,tracker.T_gw.inverse(),0.1);
            }

            glUnsetFrameOfReference();
        }

        // Process window events via GLUT
        FinishGlutFrame();
    }

    return 0;
}

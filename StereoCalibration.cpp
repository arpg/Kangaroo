#include <thread>
#include <mutex>

#include <CVars/CVar.h>

#include <SimpleGui/Gui.h>
#include <SimpleGui/GetPot>
#include <SimpleGui/GLMesh.h>

#include <RPG/Devices/Camera/CameraDevice.h>

#include <fiducials/tracker.h>
#include <fiducials/drawing.h>

using namespace std;
using namespace Eigen;

const int DESIRED_WIDTH = 320;
const int DESIRED_HEIGHT = 240;

const int WINDOW_WIDTH  = 2 * DESIRED_WIDTH;
const int WINDOW_HEIGHT = 2 * DESIRED_HEIGHT;

template<typename Cam>
struct CamParamFixed {
    const static int PARAMS = 0;

    static inline Eigen::Matrix<double,2,2> dmap_by_duv(const LinearCamera& cam, const Eigen::Vector2d&)
    {
        Eigen::Matrix<double,2,2> ret;
        ret << cam.K()(0,0), 0,
                0, cam.K()(1,1);
        return ret;
    }

    static inline Eigen::Matrix<double,PARAMS,2> dmap_by_dk(const Cam&, const Eigen::Vector2d&)
    {
        return Eigen::Matrix<double,PARAMS,2>();
    }

    static inline void UpdateCam(MatlabCamera&, const Eigen::Matrix<double,PARAMS,1>& )
    {
    }
};

template<typename Cam>
struct CamParamLinearFuEqFv {
    const static int PARAMS = 3;

    static inline Eigen::Matrix<double,2,2> dmap_by_duv(const LinearCamera& cam, const Eigen::Vector2d&)
    {
        Eigen::Matrix<double,2,2> ret;
        ret << cam.K()(0,0), 0,
                0, cam.K()(1,1);
        return ret;
    }

    static inline Eigen::Matrix<double,PARAMS,2> dmap_by_dk(const Cam& cam, const Eigen::Vector2d& x)
    {
        const double u = x(0);
        const double v = x(1);
        const double f  = cam.K()(0,0);
        const double u0 = cam.K()(0,2);
        const double v0 = cam.K()(1,2);

        Eigen::Matrix<double,PARAMS,2> d;
        d <<    f * u, f * v,
                u0, 0,
                0, v0;
        return d;
    }

    static inline void UpdateCam(MatlabCamera& cam, const Eigen::Matrix<double,PARAMS,1>& x )
    {
        Eigen::Matrix3d K = cam.K();
        K(0,0) *= exp(x(0));
        K(1,1) *= exp(x(0));
        K(0,2) *= exp(x(1));
        K(1,2) *= exp(x(2));
        cam.SetK(K);
    }
};

template<typename Cam>
struct CamParamLinear {
    const static int PARAMS = 4;

    static inline Eigen::Matrix<double,2,2> dmap_by_duv(const LinearCamera& cam, const Eigen::Vector2d&)
    {
        Eigen::Matrix<double,2,2> ret;
        ret << cam.K()(0,0), 0,
                0, cam.K()(1,1);
        return ret;
    }

    static inline Eigen::Matrix<double,PARAMS,2> dmap_by_dk(const Cam& cam, const Eigen::Vector2d& x)
    {
        const double u = x(0);
        const double v = x(1);
        const double fu = cam.K()(0,0);
        const double fv = cam.K()(1,1);
        const double u0 = cam.K()(0,2);
        const double v0 = cam.K()(1,2);

        Eigen::Matrix<double,PARAMS,2> d;
        d <<    fu * u, 0,
                0, fv * v,
                u0, 0,
                0, v0;
        return d;
    }

    static inline void UpdateCam(MatlabCamera& cam, const Eigen::Matrix<double,PARAMS,1>& x )
    {
        Eigen::Matrix3d K = cam.K();
        K(0,0) *= exp(x(0));
        K(1,1) *= exp(x(1));
        K(0,2) *= exp(x(2));
        K(1,2) *= exp(x(3));
        cam.SetK(K);
    }
};

template<typename Cam>
struct CamParamMatlab {
    const static int PARAMS = 4 + 2;

    static inline Eigen::Matrix<double,2,2> dmap_by_duv(const MatlabCamera& cam, const Eigen::Vector2d& x)
    {
        const double k1 = cam._k1;
        const double k2 = cam._k2;
        const double u = x(0);
        const double uu = u*u;
        const double uuu = uu*u;
        const double v = x(1);
        const double vv = v*v;
        const double vvv = vv*v;

        const double dpoly_by_u = 2*k1*u + k2*(4*uuu + 4*u*vv); // + k3*(6*u*vvv*v + 12*uuu*vv + 6*uuu*uu);
        const double dpoly_by_v = 2*k1*v + k2*(4*vvv + 4*v*uu); // + k3*(6*v*uuu*u + 12*vvv*uu + 6*vvv*vv);

        Eigen::Matrix<double,2,2> ret;
        ret << cam.K()(0,0) * (dpoly_by_u*u + 1), cam.K()(0,0) * dpoly_by_v * u,
                cam.K()(1,1) * dpoly_by_u*v, cam.K()(1,1) * (dpoly_by_v*v + 1);
        return ret;
    }

    static inline Eigen::Matrix<double,PARAMS,2> dmap_by_dk(const Cam& cam, const Eigen::Vector2d& x)
    {
        const double u = x(0);
        const double v = x(1);

        const double k1 = cam._k1;
        const double k2 = cam._k2;

        const double rd2 = u*u + v*v;
        const double rd4 = rd2*rd2;
//        const double rd6 = rd4*rd6;

        const double fu = cam.K()(0,0);
        const double fv = cam.K()(1,1);
        const double u0 = cam.K()(0,2);
        const double v0 = cam.K()(1,2);

        const double poly = 1 + k1*rd2 + k2*rd4; // + k3*rd6;

        Eigen::Matrix<double,PARAMS,2> d;
        d <<    fu * poly * u, 0,
                0, fv * poly * v,
                u0, 0,
                0, v0,
                fu * k1*rd2 * u, fv * k1*rd2 * v,
                fu * k2*rd4 * u, fv * k2*rd4 * v;
//                fu * k3*rd6 * u, fv * k3*rd6 * v;
        return d;
    }

    static inline void UpdateCam(MatlabCamera& cam, const Eigen::Matrix<double,PARAMS,1>& x )
    {
        Eigen::Matrix3d K = cam.K();
        K(0,0) *= exp(x(0));
        K(1,1) *= exp(x(1));
        K(0,2) *= exp(x(2));
        K(1,2) *= exp(x(3));
        cam.SetK(K);

        cam._k1 *= exp(x(4));
        cam._k2 *= exp(x(5));
//        cam._k3 *= exp(x(6));
    }
};

inline void glDrawTexturesQuad(float t, float b, float l, float r)
{
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2f(l,b);
    glTexCoord2f(1,0); glVertex2f(r,b);
    glTexCoord2f(1,1); glVertex2f(r,t);
    glTexCoord2f(0,1); glVertex2f(l,t);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

//CVarUtils::CreateCVar<ConsoleFunc>("TestConsoleFunc",&TestConsoleFunc);
//bool TestConsoleFunc( std::vector<std::string> *args)
//{
//    for(int i=0; i<args->size(); ++i ) {
//        cout << args->at(i) << endl;
//    }
//}

inline bool Pushed(bool& button)
{
    const bool pushed = button;
    button = false;
    return pushed;
}

struct StereoKeyframe
{
    Sophus::SE3 T_fw[2];
    Eigen::Matrix<double,2,Eigen::Dynamic> obs[2];
};

template<typename T, unsigned NC, unsigned NJ0, unsigned NJ1>
inline void AddSparseOuterProduct(
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& JTJ,
        Eigen::Matrix<T,Eigen::Dynamic,1>& JTy,
        const Eigen::Matrix<T,NJ0,NC>& J0, int J0pos,
        const Eigen::Matrix<T,NJ1,NC>& J1, int J1pos,
        const Eigen::Matrix<T,NC,1>& err
) {
    // On diagonal blocks
    if(NJ0) JTJ.template block<NJ0,NJ0>(J0pos,J0pos) += J0 * J0.transpose();
    if(NJ1) JTJ.template block<NJ1,NJ1>(J1pos,J1pos) += J1 * J1.transpose();

    // Lower diagonal blocks
    if(NJ0 && NJ1) JTJ.template block<NJ1,NJ0>(J1pos,J0pos) += J1 * J0.transpose();

    // Upper diagonal blocks TODO: use only one diagonal in future
    if(NJ0 && NJ1) JTJ.template block<NJ0,NJ1>(J0pos,J1pos) += J0 * J1.transpose();

    // Errors
    for(int i=0; i<NC; ++i) {
        if(NJ0) JTy.template segment<NJ0>(J0pos) += J0.col(i) * err(i);
        if(NJ1) JTy.template segment<NJ1>(J1pos) += J1.col(i) * err(i);
    }
}

template<typename T, unsigned NC, unsigned NJ0, unsigned NJ1, unsigned NJ2 >
inline void AddSparseOuterProduct(
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& JTJ,
        Eigen::Matrix<T,Eigen::Dynamic,1>& JTy,
        const Eigen::Matrix<T,NJ0,NC>& J0, int J0pos,
        const Eigen::Matrix<T,NJ1,NC>& J1, int J1pos,
        const Eigen::Matrix<T,NJ2,NC>& J2, int J2pos,
        const Eigen::Matrix<T,NC,1>& err
) {
    // On diagonal blocks
    if(NJ0) JTJ.template block<NJ0,NJ0>(J0pos,J0pos) += J0 * J0.transpose();
    if(NJ1) JTJ.template block<NJ1,NJ1>(J1pos,J1pos) += J1 * J1.transpose();
    if(NJ2) JTJ.template block<NJ2,NJ2>(J2pos,J2pos) += J2 * J2.transpose();

    // Lower diagonal blocks
    if(NJ1 && NJ0) JTJ.template block<NJ1,NJ0>(J1pos,J0pos) += J1 * J0.transpose();
    if(NJ2 && NJ0) JTJ.template block<NJ2,NJ0>(J2pos,J0pos) += J2 * J0.transpose();
    if(NJ2 && NJ1) JTJ.template block<NJ2,NJ1>(J2pos,J1pos) += J2 * J1.transpose();

    // Upper diagonal blocks TODO: use only one diagonal in future
    if(NJ0 && NJ1) JTJ.template block<NJ0,NJ1>(J0pos,J1pos) += J0 * J1.transpose();
    if(NJ0 && NJ2) JTJ.template block<NJ0,NJ2>(J0pos,J2pos) += J0 * J2.transpose();
    if(NJ1 && NJ2) JTJ.template block<NJ1,NJ2>(J1pos,J2pos) += J1 * J2.transpose();

    // Errors
    for(int i=0; i<NC; ++i) {
        if(NJ0) JTy.template segment<NJ0>(J0pos) += J0.col(i) * err(i);
        if(NJ1) JTy.template segment<NJ1>(J1pos) += J1.col(i) * err(i);
        if(NJ2) JTy.template segment<NJ2>(J2pos) += J2.col(i) * err(i);
    }
}

static inline Eigen::Matrix<double,2,3> dpi_dx(const Eigen::Vector3d& x)
{
    const double x2x2 = x(2)*x(2);
    Eigen::Matrix<double,2,3> ret;
    ret << 1.0 / x(2), 0,  -x(0) / x2x2,
            0, 1.0 / x(2), -x(1) / x2x2;
    return ret;
}

static inline Eigen::Matrix<double,4,4> se3_gen(unsigned i) {

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

static void OptimiseIntrinsicsPoses(
    Eigen::Matrix<double,3,Eigen::Dynamic> pattern,
    std::vector<StereoKeyframe>& keyframes,
    MatlabCamera& cam,
    Sophus::SE3& T_rl
) {
    // keyframes.size might increase asynchronously, so save
    const int N = keyframes.size();

    typedef CamParamMatlab<MatlabCamera> CamParam;
    const int PARAMS_K = CamParam::PARAMS;
    const int PARAMS_T = 6;
    const int PARAMS_TOTAL = PARAMS_K + (1+N)* PARAMS_T;

    unsigned int num_obs = 0;
    double sumsqerr = 0;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> JTJ(PARAMS_TOTAL,PARAMS_TOTAL);
    Eigen::Matrix<double,Eigen::Dynamic,1> JTy(PARAMS_TOTAL);
    JTJ.setZero();
    JTy.setZero();

    // Make JTJ and JTy from observations over each Keyframe
    for( size_t kf=0; kf < N; ++kf ) {
        // For each observation
        for( size_t on=0; on < pattern.cols(); ++on ) {
            // Construct block contributions for JTJ and JTy
            const Sophus::SE3 T_lt = keyframes[kf].T_fw[0];

            const Eigen::Vector3d Pt = pattern.col(on);
            const Eigen::Vector3d Pl = T_lt * Pt;

            const Eigen::Vector2d obsl = keyframes[kf].obs[0].col(on);
            if( isfinite(obsl[0]) ) {
                const Eigen::Vector2d pl_ = project(Pl);
                const Eigen::Vector2d pl = cam.map(pl_);
                const Eigen::Vector2d errl = pl - obsl;
                sumsqerr += errl.squaredNorm();
                num_obs++;

                Eigen::Matrix<double,PARAMS_K,2> Jk = CamParam::dmap_by_dk(cam,pl_);

                const Eigen::Matrix<double,2,3> dpi = dpi_dx(Pl);
                const Eigen::Matrix<double,2,2> dmap = CamParam::dmap_by_duv(cam,pl_);
                const Eigen::Matrix<double,2,3> dmapdpi = dmap * dpi;
                const Eigen::Matrix<double,2,4> dmapdpiTlt = dmapdpi * T_lt.matrix3x4();

                Eigen::Matrix<double,PARAMS_T,2> J_T_lw;
                for(int i=0; i<PARAMS_T; ++i ) {
                    J_T_lw.row(i) = dmapdpiTlt * se3_gen(i) * unproject(Pt);
                }

                AddSparseOuterProduct<double,2,PARAMS_K,PARAMS_T>(
                    JTJ,JTy,  Jk,0,  J_T_lw,PARAMS_K+(1+kf)*PARAMS_T, errl
                );
            }

            const Eigen::Vector2d obsr = keyframes[kf].obs[1].col(on);
            if(isfinite(obsr[0])) {
                const Eigen::Vector3d Pr = T_rl * Pl;
                const Eigen::Vector2d pr_ = project(Pr);
                const Eigen::Vector2d pr  = cam.map(pr_);
                const Eigen::Vector2d errr = pr - obsr;
                sumsqerr += errr.squaredNorm();
                num_obs++;

                Eigen::Matrix<double,PARAMS_K,2> Jk = CamParam::dmap_by_dk(cam,pr_);

                const Eigen::Matrix<double,2,3> dpi = dpi_dx(Pr);
                const Eigen::Matrix<double,2,2> dmap = CamParam::dmap_by_duv(cam,pr_);
                const Eigen::Matrix<double,2,3> dmapdpi = dmap * dpi;
                const Eigen::Matrix<double,2,4> dmapdpiT_rl = dmapdpi * T_rl.matrix3x4();

                Eigen::Matrix<double,PARAMS_T,2> J_T_rl;
                Eigen::Matrix<double,PARAMS_T,2> J_T_lw;
                for(int i=0; i<PARAMS_T; ++i ) {
                    J_T_rl.row(i) = dmapdpiT_rl * se3_gen(i) * T_lt.matrix() * unproject(Pt);
                    J_T_lw.row(i) = dmapdpiT_rl * T_lt.matrix() * se3_gen(i) * unproject(Pt);
                }

                AddSparseOuterProduct<double,2,PARAMS_K,PARAMS_T,PARAMS_T>(
                    JTJ,JTy,  Jk,0,  J_T_rl,PARAMS_K,  J_T_lw,PARAMS_K+(1+kf)*PARAMS_T, errr
                );
            }
        }
    }

    cout << "=============== RMSE: " << sqrt(sumsqerr/num_obs) << " ====================" << endl;

    FullPivLU<MatrixXd> lu_JTJ(JTJ);
    Eigen::Matrix<double,Eigen::Dynamic,1> x = -1.0 * lu_JTJ.solve(JTy);

    if( x.norm() > 1 ) {
        x = x / x.norm();
    }

    if( lu_JTJ.rank() == PARAMS_TOTAL )
    {
        CamParam::UpdateCam(cam, x.head<PARAMS_K>());

        // Update baseline
        T_rl = T_rl * Sophus::SE3::exp(x.segment<PARAMS_T>(PARAMS_K) );

        // Update poses
        for( size_t kf=0; kf < N; ++kf ) {
            keyframes[kf].T_fw[0] = keyframes[kf].T_fw[0] *
                Sophus::SE3::exp(x.segment<PARAMS_T>(PARAMS_K + (1+kf)*PARAMS_T));
            keyframes[kf].T_fw[1] = T_rl * keyframes[kf].T_fw[0];
        }

        cout << cam << endl;
        cout << T_rl.matrix() << endl;
    }else{
        cerr << "Rank deficient! Missing: " << (PARAMS_TOTAL - lu_JTJ.rank()) << endl;
        cerr << lu_JTJ.kernel() << endl;
    }
}


class Application
{
public:
    Application()
        : window(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, __FILE__ ),
          shouldQuit(false)
    {
        Init();
    }

    ~Application()
    {
    }

    void InitCamera()
    {
        const int devid = 0;

        // Setup Camera device
        if(devid == 0) {
            camera.SetProperty<int>("NumImages", 2);
            camera.SetProperty<int>("ImageWidth", 640);
            camera.SetProperty<int>("ImageHeight", 480);
//            camera.SetProperty<double>("FPS", 15);
            camera.SetProperty<int>("BufferCount", 60);

            if( !camera.InitDriver( "Dvi2Pci" ) ) {
                std::cerr << "Failed to init Dvi2Pci." << std::endl;
                exit(0);
            }
        }else if(devid == 1) {
            // Setup Camera
            camera.SetProperty("StartFrame",    0);
            camera.SetProperty("DataSourceDir", "/home/slovegrove/data/CityBlock-Noisy" );
            camera.SetProperty("Channel-0",     "left.*pgm" );
            camera.SetProperty("Channel-1",     "right.*pgm" );
            camera.SetProperty("NumChannels",   2 );
            camera.InitDriver("FileReader");
        }else if(devid == 2){
            camera.SetProperty("NumChannels", 2 );
            camera.SetProperty("CamUUID0", 5004955);
            camera.SetProperty("CamUUID1", 5004954);
            camera.SetProperty("ImageBinningX", 2);
            camera.SetProperty("ImageBinningY", 2);
            camera.SetProperty("ImageWidth", 694);
            camera.SetProperty("ImageHeight", 518);
            if(!camera.InitDriver( "AlliedVision" )) {
                cerr << "Couldn't start driver for camera " << endl;
                exit(1);
            }
        }else{
            exit(0);
        }
        camera.Capture(img);
        width = img[0].width();
        height = img[0].height();

        // Setup camera parameters
        VectorXd camParamsVec(6); // = Var<Matrix<double,9,1> >("cam_params");
        camParamsVec << 0.558526, 0.747774, 0.484397, 0.494393, -0.249261, 0.0825967;
        camParams = MatlabCamera( width,height, camParamsVec);

        // Setup stereo baseline
        Eigen::Matrix3d R_rl;
        R_rl << 0.999995,   0.00188482,  -0.00251896,
                -0.0018812,     0.999997,   0.00144025,
                0.00252166,  -0.00143551,     0.999996;

        Eigen::Vector3d l_r;
        l_r <<    -0.203528, -0.000750334, 0.00403201;

        T_rl = Sophus::SE3(R_rl, l_r);
    }

    void InitOpenGLTextures()
    {
        // Create two OpenGL textures for stereo images
        glGenTextures(2, m_glTex);

        // Allocate texture memory on GPU
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        glPixelStorei(GL_PACK_ALIGNMENT,1);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    }

    void InitTrackers()
    {
        // Setup Tracker objects
        // Unit hell!
        const double ppi = 72; // Points Per Inch
        const double USwp = 11 * ppi;
        const double UShp = 8.5 * ppi;
        const double mpi = 0.0254; // meters per inch
        const double mpp = mpi / ppi; // meters per point
        const double unit = mpp; //1; //mpp;

        // Setup Trackers
        for(int i=0; i<2; ++i) {
            tracker[i] = new Tracker(width,height);
            tracker[i]->target.GenerateRandom(60,unit*USwp*25/(842.0),unit*USwp*75/(842.0),unit*USwp*40/(842.0),Eigen::Vector2d(unit*USwp,unit*UShp));
//            tracker[i]->target.SaveEPS("stereo.eps");
        }
    }

    void Init()
    {
        InitCamera();
        InitOpenGLTextures();
        InitTrackers();

        // Setup floor grid
        window.AddChildToRoot(new GLGrid());

        // Setup OpenGL Render Callback
        window.AddPostRenderCallback( Application::PostRender, this);

    }

    static void PostRender(GLWindow*, void* data)
    {
        Application* self = (Application*)data;
        self->Draw();
    }

    void OptimiseRun()
    {
        Eigen::Matrix<double,3,Eigen::Dynamic> pattern = tracker[0]->TargetPattern3D();

        while(!shouldQuit) {
            if(keyframes.size() > 2 ) {
                OptimiseIntrinsicsPoses(pattern,keyframes,camParams, T_rl);
            }else{
                usleep(1000);
            }
        }
    }

    void CameraRun()
    {
        bool& save_kf = CVarUtils::CreateCVar<bool>( "SaveKeyframe", false );

        std::thread trackerThreads[2];

        while(!shouldQuit) {
            camera.Capture(img);

            for(int i=0; i<2; ++i) {
                trackerThreads[i] = std::thread(boost::bind(&Tracker::ProcessFrame, tracker[i], camParams,img[i].Image.data) );
            }
            for(int i=0; i<2; ++i) {
                trackerThreads[i].join();
            }

            if( Pushed(save_kf) ) {
                cout << "Save Keyframe" << endl;

                if(keyframes.size() == 0) {
                    // First keyframe. Initialise baseline estimate
                    T_rl = tracker[1]->T_hw * tracker[0]->T_hw.inverse();
                }

                StereoKeyframe kf;
                for(int i=0; i<2; ++i) {
                    kf.obs[i] = tracker[i]->TargetPatternObservations();
                    kf.T_fw[i] = tracker[i]->T_hw;
                }
                kfMutex.lock();
                keyframes.push_back(kf);
                kfMutex.unlock();
            }
        }
    }

    int Run()
    {
        // Run Camera Loop
        std::thread camThread( std::bind( &Application::CameraRun, this ) );

        // Run Optimisation Loop
        std::thread optThread( std::bind( &Application::OptimiseRun, this ) );

        // Run GUI
        window.Run();
        shouldQuit = true;

        camThread.join();
    }

    void Draw()
    {
        glClearColor (0.0, 0.0, 0.0, 0.0);
//        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_LIGHTING);
        glDisable(GL_COLOR_MATERIAL );

        // Draw 3D stuff using the ModelView Matrix from SimpleGUI
        glViewport(0,0,DESIRED_WIDTH,DESIRED_HEIGHT);
        DrawTarget(tracker[0]->target,Eigen::Vector2d(0,0),1,0.2,0.2);
        for(int i=0; i<2; ++i) {
            glSetFrameOfReferenceF(tracker[i]->T_hw.inverse());
            glDrawAxis(0.2);
            glUnsetFrameOfReference();
        }

        // Draw Stereo keyframes
        for(size_t kf=0; kf < keyframes.size(); ++kf ) {
            glSetFrameOfReferenceF(keyframes[kf].T_fw[0].inverse());
            glDrawAxis(0.2);
            glUnsetFrameOfReference();

            glSetFrameOfReferenceF(keyframes[kf].T_fw[1].inverse());
            glDrawAxis(0.2);
            glUnsetFrameOfReference();
        }

        // Reset model view matrices for displaying images
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        // Draw Threshold images for cam 0
        glViewport(DESIRED_WIDTH,0,DESIRED_WIDTH,DESIRED_HEIGHT);
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_LUMINANCE,GL_UNSIGNED_BYTE,tracker[0]->tI.get());
        glColor3f (1.0, 1.0, 1.0);
        glDrawTexturesQuad(-1,1,-1,1);

        // Upload textures for images
        if( img.size() >= 2 ) {
            glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_LUMINANCE,GL_UNSIGNED_BYTE,img[0].Image.data);
            glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_LUMINANCE,GL_UNSIGNED_BYTE,img[1].Image.data);
        }

        glOrtho(-0.5,width-0.5,height-0.5,-0.5,0,1.0);

        // Draw left image
        glViewport(0,DESIRED_HEIGHT,DESIRED_WIDTH,DESIRED_HEIGHT);
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glColor3f (1.0, 1.0, 1.0);
        glDrawTexturesQuad(height,0,0,width);
        for( int i=0; i<tracker[0]->conics.size(); ++i ) {
          glBinColor(tracker[0]->conics_target_map[i],tracker[0]->target.circles3D().size());
          DrawCross(tracker[0]->conics[i].center,2);
        }
//        glDrawTexturesQuad(-1,1,-1,1);

        // Draw right image
        glViewport(DESIRED_WIDTH,DESIRED_HEIGHT,DESIRED_WIDTH,DESIRED_HEIGHT);
        glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
        glColor3f (1.0, 1.0, 1.0);
        glDrawTexturesQuad(height,0,0,width);
        for( int i=0; i<tracker[1]->conics.size(); ++i ) {
          glBinColor(tracker[1]->conics_target_map[i],tracker[1]->target.circles3D().size());
          DrawCross(tracker[1]->conics[i].center,2);
        }

        glDisable(GL_TEXTURE_2D);

        // Reset OpenGL to what SimpleGui expects
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
//        glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);
        glViewport(0,0,DESIRED_WIDTH,DESIRED_HEIGHT);

    }

    GLWindow window;
    CameraDevice camera;

    std::vector<rpg::ImageWrapper> img;
    GLuint m_glTex[2];
    int width, height;
    std::mutex kfMutex;

    MatlabCamera camParams;
    Tracker* tracker[2];

    Sophus::SE3 T_rl;
    std::vector<StereoKeyframe> keyframes;

    bool shouldQuit;
};

int main (int argc, char** argv){
    Application app;
    return app.Run();
}

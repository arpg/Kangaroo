#include "common/ViconTracker.h"
#include "common/BaseDisplayCuda.h"
#include "common/RpgCameraOpen.h"
#include "common/Handler3dGpuDepth.h"
#include "common/ImageSelect.h"
#include "common/CameraModelPyramid.h"
#include "common/CVarHelpers.h"
#include <CVars/CVar.h>

#include <boost/ptr_container/ptr_map.hpp>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include <kangaroo/kangaroo.h>

#include <SceneGraph/SceneGraph.h>
#include <unsupported/Eigen/OpenGLSupport>

using namespace std;

const std::string ViconIp = "192.168.10.1";
const int MaxWidth = 640;
const int MaxHeight = 480;
const int w = MaxWidth;
const int h = MaxHeight;

namespace CVarUtils {
    inline std::ostream& operator<<( std::ostream& Stream, const SceneGraph::GLObject& object )
    {
        CVarUtils::operator <<(Stream, object.GetPose());
        return Stream;
    }

    inline std::istream& operator>>( std::istream& Stream, SceneGraph::GLObject& object )
    {
        Eigen::Matrix<double,6,1> pose;
        CVarUtils::operator >>(Stream, pose);
        object.SetPose(pose);
        return Stream;
    }
}

struct Sensor {
    Sensor(std::string name, ViconConnection& viconSharedConnection, int w, int h)
        :name(name), tracker(name, viconSharedConnection), w(w), h(h),
          vbo(pangolin::GlArrayBuffer, w,h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW ),
          cbo(pangolin::GlArrayBuffer, w,h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW )
    {
        K.u0 = w/2.0f;
        K.v0 = h/2.0f;

        glT_wv = new SceneGraph::GLAxis(0.1);
        glT_vs = new SceneGraph::GLMovableAxis(0.001,false,false);
        glplane = new SceneGraph::GLGrid(10,1,false);
        glxytheta = new SceneGraph::GLWayPoint();
        glxytheta->ClampToPlane(Eigen::Vector4d(0,0,1,0));

//        glT_vs->SetPose(SceneGraph::GLCart2T(x,y,z,p,q,r));
        glvbo  = new SceneGraph::GLVbo(&vbo,0, &cbo);
        glT_wv->AddChild(glT_vs);
        glT_vs->AddChild(glvbo);
//        glT_vs->AddChild(glplane);

        nd_c << 0, 0, -1;

        CVarUtils::AttachCVar<SceneGraph::GLObject>("pose_"+name, glxytheta);
        CVarUtils::AttachCVar<Eigen::Vector3d>("plane_"+name, &nd_c);

    }

    std::string name;
    ViconTracking tracker;
    int w, h;
    pangolin::GlBufferCudaPtr vbo;
    pangolin::GlBufferCudaPtr cbo;
    Gpu::ImageIntrinsics K;

    Eigen::Vector3d nd_c;

    SceneGraph::GLAxis* glT_wv;
    SceneGraph::GLMovableAxis* glT_vs;
    SceneGraph::GLVbo* glvbo;
    SceneGraph::GLGrid* glplane;
    SceneGraph::GLWayPoint* glxytheta;
};

typedef boost::ptr_map<std::string,Sensor> SensorPtrMap;

int main( int argc, char* argv[] )
{
    pangolin::View& container = SetupPangoGLWithCuda(1024, 768, 180, __FILE__);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();
    glDisable( GL_LIGHTING );
    glDisable( GL_COLOR_MATERIAL);

    CameraDevice video = OpenRpgCamera(argc,argv,1);
    std::vector<rpg::ImageWrapper> images;

    Gpu::Image<unsigned short, Gpu::TargetDevice, Gpu::Manage> imgd(MaxWidth,MaxHeight);
    Gpu::Image<float,  Gpu::TargetDevice, Gpu::Manage> imgf(MaxWidth,MaxHeight);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> imgv(MaxWidth,MaxHeight);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> imgn(MaxWidth,MaxHeight);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage>  rayi(MaxWidth,MaxHeight);
    Gpu::Image<float, Gpu::TargetDevice, Gpu::Manage>  rayd(MaxWidth,MaxHeight);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> rayn(MaxWidth,MaxHeight);
    Gpu::Image<unsigned char,Gpu::TargetDevice,Gpu::Manage> scratch(MaxWidth*sizeof(Gpu::LeastSquaresSystem<float,6>),MaxHeight);
    Gpu::Image<float,Gpu::TargetDevice,Gpu::Manage>  imgerr(w,h);

    Gpu::BoundedVolume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(4*64,4*64,2*64, make_float3(-2,-2,-0.2), make_float3(2,2,1.8));
    const float3 voxsize = vol.VoxelSizeUnits();

    SceneGraph::GLSceneGraph graph;
    SceneGraph::GLGrid glGrid(10,1,true);
    SceneGraph::GLAxis glAxis;
    SceneGraph::GLAxisAlignedBox glbbox;
    glbbox.SetBounds(Gpu::ToEigen(vol.bbox.Min()), Gpu::ToEigen(vol.bbox.Max()) );
    graph.AddChild(&glGrid);
    graph.AddChild(&glAxis);
    graph.AddChild(&glbbox);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrixRDF_TopLeft(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAtRDF(0,5,5,0,0,0,0,0,1)
    );

    SetupContainer(container, 3, (float)MaxWidth / (float)MaxHeight);
    Handler3DGpuDepth handler(rayd, s_cam);
    pangolin::ActivateDrawImage<float> adrayi(rayi, GL_LUMINANCE32F_ARB, true, true);
    pangolin::ActivateDrawImage<float4> adrayn(rayn, GL_RGBA32F, true, true);
    container[0].SetDrawFunction(boost::ref(adrayi)).SetHandler(&handler);
    container[1].SetDrawFunction(boost::ref(adrayn)).SetHandler(&handler);
    container[2].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, s_cam))
                .SetHandler( new SceneGraph::HandlerSceneGraph(graph,s_cam) );

    SensorPtrMap sensors;

    ViconConnection vicon(ViconIp);

    pangolin::Var<int> only("ui.only",-1, -1, 3);
    pangolin::Var<bool> fuse("ui.fuse", false, true);
    pangolin::Var<bool> fuseonce("ui.fuse_once", false, false);

    pangolin::Var<bool> sdfreset("ui.reset", false, false);
    pangolin::Var<bool> sdfsphere("ui.sphere", false, false);

    pangolin::Var<int> biwin("ui.size",10, 1, 20);
    pangolin::Var<float> bigs("ui.gs",10, 1E-3, 5);
    pangolin::Var<float> bigr("ui.gr",700, 1E-3, 200);

    pangolin::Var<float> trunc_dist("ui.trunc_dist", 2*length(voxsize), 2*length(voxsize),0.5);
    pangolin::Var<float> max_w("ui.max_w", 10, 1E-4, 10);
    pangolin::Var<float> mincostheta("ui.min_cos_theta", 0.5, 0, 1);

    pangolin::Var<bool> plane_do("ui.Compute_Ground_Plane", false, true);
    pangolin::Var<float> plane_maxz("ui.Plane_Within",20, 0.1, 100);
    pangolin::Var<float> plane_c("ui.Plane_c", 0.5, 0.0001, 1);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        if(Pushed(sdfreset) || frame==0) {
            Gpu::SdfReset(vol, trunc_dist);
        }

        if(Pushed(sdfsphere)) {
            Gpu::SdfSphere(vol, make_float3(0,0,1), 0.9 );
        }

        if(plane_do) {
            int ni = 0;
            for(SensorPtrMap::iterator si= sensors.begin(); si != sensors.end(); si++) {
                if(only >-1 && only != ni++) continue;
                Sensor& sensor = *si->second;
                pangolin::CudaScopedMappedPtr var(sensor.vbo);
                Gpu::Image<float4> imgvbo((float4*)*var, sensor.w, sensor.h);

                Eigen::Matrix3d U; U << sensor.w, 0, sensor.w,  sensor.h/2, sensor.h, sensor.h,  1, 1, 1;
                Eigen::Matrix3d Q = -(sensor.K.InverseMatrix() * U).transpose();
                Eigen::Matrix3d Qinv = Q.inverse();
                Eigen::Vector3d plane_invz = Q * sensor.nd_c;

                for(int i=0; i<5; ++i )
                {
                    Gpu::LeastSquaresSystem<float,3> lss = Gpu::PlaneFitGN(imgvbo, Qinv, plane_invz, scratch, imgerr, 0.2, plane_maxz, plane_c);
                    Eigen::FullPivLU<Eigen::Matrix3d> lu_JTJ( (Eigen::Matrix3d)lss.JTJ );
                    Eigen::Vector3d x = -1.0 * lu_JTJ.solve( (Eigen::Vector3d)lss.JTy );
                    if( x.norm() > 1 ) x = x / x.norm();
                    for(int i=0; i<3; ++i ) {
                        plane_invz(i) *= exp(x(i));
                    }
                    sensor.nd_c = Qinv * plane_invz;
                    sensor.glplane->SetPlane(sensor.nd_c);
                }
            }
        }

        if(video.Capture(images)) {
            for(int ni=0; ni < images.size(); ni++) {
                if(only >-1 && only != ni) continue;

                // Which camera did this come from
                std::string sensor_name = images[ni].Map.GetProperty("DeviceName", "Local" + boost::lexical_cast<std::string>(ni) );

                // Size of depthmap that this sensor will produce
                const int dw = images[ni].Image.cols;
                const int dh = images[ni].Image.rows;

                SensorPtrMap::iterator it = sensors.find(sensor_name);
                if( it == sensors.end() ) {
                    // create sensor
                    cout << "New sensor - " << sensor_name << " (" << dw << "x" << dh << ")" << endl;

                    // Create and Insert into map of active sensors
                    Sensor* newsensor = new Sensor(sensor_name, vicon, dw,dh);
                    it = sensors.insert(sensor_name, newsensor ).first;

                    newsensor->K.fu = video.GetProperty<float>("Depth" +  boost::lexical_cast<std::string>(ni) + "FocalLength",0);
                    newsensor->K.fv = newsensor->K.fu;

                    // add to posegraph
                    graph.AddChild(newsensor->glT_wv);
                    graph.AddChild(newsensor->glxytheta);

                    CVarUtils::Load("cvars.xml");
                }

                Sensor& sensor = *it->second;

//                // Set from vicon
//                sensor.glT_wv->SetPose(sensor.tracker.T_wf().matrix());

                // Set based on waypoint
                double d_c = 1.0 / sensor.nd_c.norm();
                Eigen::Vector3d n_c = d_c * sensor.nd_c;
                sensor.glT_vs->SetPose(0,0,0,0,0,0);
                Eigen::Matrix4d T_wc = sensor.glxytheta->GetPose4x4_po();
                T_wc(2,3) = d_c;
                Eigen::Matrix4d T_cc = Eigen::Matrix4d::Identity();
                T_cc.block<3,3>(0,0) = SceneGraph::Rotation_a2b(n_c, Eigen::Vector3d(0,0,1) );
                T_wc = T_wc * T_cc;
                sensor.glT_wv->SetPose(T_wc);

                // Update depth map. (lets assume we have a kinect for now)
                imgd.CopyFrom<Gpu::TargetHost,Gpu::DontManage>(images[ni].Image);
                Gpu::BilateralFilter<float,unsigned short>(imgf,imgd,bigs,bigr,biwin,200);
                Gpu::ElementwiseScaleBias<float,float,float>(imgf,imgf, 1.0f/1000.0f);
                Gpu::DepthToVbo(imgv, imgf, sensor.K );
                Gpu::NormalsFromVbo(imgn, imgv);

                if(Pushed(fuseonce) || fuse) {
                    // integrate gtd into TSDF
                    Eigen::Matrix<double,3,4> T_cw = (Sophus::SE3(sensor.glT_wv->GetPose4x4_po() * sensor.glT_vs->GetPose4x4_po())).inverse().matrix3x4();
                    Gpu::SdfFuse(vol, imgf, imgn, T_cw, sensor.K, trunc_dist, max_w, mincostheta );
                }

                {
                    pangolin::CudaScopedMappedPtr var(sensor.vbo);
                    Gpu::Image<float4> dVbo((float4*)*var,dw,dh);
                    dVbo.CopyFrom(imgv);
                }

                {
                    pangolin::CudaScopedMappedPtr var(sensor.cbo);
                    Gpu::Image<float4> dCbo((float4*)*var,dw,dh);
                    dCbo.CopyFrom(imgn);
                }
            }
        }

        // Raycast current view
        {
            Sophus::SE3 T_vw(s_cam.GetModelViewMatrix());
            Gpu::RaycastSdf(rayd, rayn, rayi, vol, T_vw.inverse().matrix3x4(), Gpu::ImageIntrinsics(420,420,320,320), 0.1, 1000, true );
        }

        int sn = 0;
        for(SensorPtrMap::iterator si= sensors.begin(); si != sensors.end(); si++) {
            Sensor& sensor = *si->second;
            sensor.glT_wv->SetVisible(only==-1 || only==sn);
            sn++;
        }


        // Render stuff
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
        pangolin::FinishGlutFrame();
    }
}

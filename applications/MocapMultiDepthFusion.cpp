#include "common/ViconTracker.h"
#include "common/BaseDisplayCuda.h"
#include "common/RpgCameraOpen.h"
#include "common/Handler3dGpuDepth.h"
#include "common/ImageSelect.h"

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

struct Sensor {
    Sensor(std::string name, ViconConnection& viconSharedConnection, int w, int h)
        :name(name), tracker(name, viconSharedConnection), w(w), h(h),
          vbo(pangolin::GlArrayBuffer, w,h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW ),
          cbo(pangolin::GlArrayBuffer, w,h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW )
    {
        u0 = w/2.0f;
        v0 = h/2.0f;
        ifstream ifs(name + "_f_T_vc.txt");
        ifs >> fu;
        fv = fu;
        double x,y,z,p,q,r;
        ifs >> x;
        ifs >> y;
        ifs >> z;
        ifs >> p;
        ifs >> q;
        ifs >> r;

        glT_wv = new SceneGraph::GLAxis(0.1);
        glT_vs = new SceneGraph::GLMovableAxis();
        glT_vs->SetPose(SceneGraph::GLCart2T(x,y,z,p,q,r));
        glvbo  = new SceneGraph::GLVbo(&vbo,0, &cbo);
        glT_wv->AddChild(glT_vs);
        glT_vs->AddChild(glvbo);
    }

    std::string name;
    ViconTracking tracker;
    int w, h;
    pangolin::GlBufferCudaPtr vbo;
    pangolin::GlBufferCudaPtr cbo;
    float fu,fv,u0,v0;

    SceneGraph::GLAxis* glT_wv;
    SceneGraph::GLMovableAxis* glT_vs;
    SceneGraph::GLVbo* glvbo;
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

    const int res = 256;
    Gpu::Volume<Gpu::SDF_t, Gpu::TargetDevice, Gpu::Manage> vol(res,res,res);
    const float3 boxmax = make_float3(1,1,2);
    const float3 boxmin = make_float3(-1,-1,0);
    const float3 boxsize = boxmax - boxmin;
    const float3 voxsize = boxsize / make_float3(vol.w, vol.h, vol.d);
    Gpu::SdfSphere(vol, boxmin, boxmax, make_float3(0,0,1), 0.9 );

    SceneGraph::GLSceneGraph graph;
    SceneGraph::GLGrid glGrid(10,1,false);
    SceneGraph::GLAxis glAxis;
    SceneGraph::GLAxisAlignedBox glbbox;
    glbbox.SetBounds(boxmin.x,boxmin.y,boxmin.z,boxmax.x,boxmax.y,boxmax.z);
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

    pangolin::Var<int> only("ui.only",-1, -1, 2);
    pangolin::Var<bool> fuse("ui.fuse", false, true);
    pangolin::Var<bool> fuseonce("ui.fuse once", false, false);

    pangolin::Var<bool> sdfreset("ui.reset", false, false);
    pangolin::Var<bool> sdfsphere("ui.sphere", false, false);

    pangolin::Var<int> biwin("ui.size",10, 1, 20);
    pangolin::Var<float> bigs("ui.gs",10, 1E-3, 5);
    pangolin::Var<float> bigr("ui.gr",700, 1E-3, 200);

    pangolin::Var<float> trunc_dist("ui.trunc dist", 2*length(voxsize), 2*length(voxsize),0.5);
    pangolin::Var<float> max_w("ui.max w", 10, 1E-4, 10);
    pangolin::Var<float> mincostheta("ui.min cos theta", 0.5, 0, 1);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        if(Pushed(sdfreset)) {
            Gpu::SdfReset(vol, trunc_dist);
        }

        if(Pushed(sdfsphere)) {
            Gpu::SdfSphere(vol, boxmin, boxmax, make_float3(0,0,1), 0.9 );
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

                    // add to posegraph
                    graph.AddChild(newsensor->glT_wv);
                }

                Sensor& sensor = *it->second;

                sensor.glT_wv->SetPose(sensor.tracker.T_wf().matrix());

                // Update depth map. (lets assume we have a kinect for now)
                imgd.CopyFrom<Gpu::TargetHost,Gpu::DontManage>(images[ni].Image);
                Gpu::BilateralFilter<float,unsigned short>(imgf,imgd,bigs,bigr,biwin,200);
                Gpu::ElementwiseScaleBias<float,float,float>(imgf,imgf, 1.0f/1000.0f);
                Gpu::DepthToVbo(imgv, imgf, sensor.fu, sensor.fv, sensor.u0, sensor.v0 );
                Gpu::NormalsFromVbo(imgn, imgv);

                if(Pushed(fuseonce) || fuse) {
                    // integrate gtd into TSDF
                    Eigen::Matrix<double,3,4> T_cw = (sensor.tracker.T_wf() * Sophus::SE3(sensor.glT_vs->GetPose4x4_po())).inverse().matrix3x4();
                    Gpu::SdfFuse(vol, boxmin, boxmax, imgf, imgn, T_cw, sensor.fu, sensor.fv, sensor.u0, sensor.v0, trunc_dist, max_w, mincostheta );
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
            Gpu::RaycastSdf(rayd, rayn, rayi, vol, boxmin, boxmax, T_vw.inverse().matrix3x4(), 420,420,320,320, 0.1, 1000, trunc_dist, true );
        }

        // Render stuff
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
        pangolin::FinishGlutFrame();
    }
}

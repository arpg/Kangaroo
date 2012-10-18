#include "common/ViconTracker.h"
#include "common/BaseDisplayCuda.h"
#include "common/RpgCameraOpen.h"

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

    SceneGraph::GLSceneGraph graph;
    SceneGraph::GLGrid glGrid(10,1,false);
    SceneGraph::GLAxis glAxis;
    graph.AddChild(&glGrid);
    graph.AddChild(&glAxis);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAt(0,5,5,0,0,0,0,0,1)
    );

    SetupContainer(container, 1, 640.0f/480.0f);
    container[0].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, s_cam))
                .SetHandler( new SceneGraph::HandlerSceneGraph(graph,s_cam, pangolin::AxisZ) );

    SensorPtrMap sensors;

    CameraDevice video = OpenRpgCamera(argc,argv,1);
    std::vector<rpg::ImageWrapper> images;

    Gpu::Image<unsigned short, Gpu::TargetDevice, Gpu::Manage> imgd(MaxWidth,MaxHeight);
    Gpu::Image<float,  Gpu::TargetDevice, Gpu::Manage> imgf(MaxWidth,MaxHeight);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> imgv(MaxWidth,MaxHeight);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> imgn(MaxWidth,MaxHeight);

    ViconConnection vicon(ViconIp);

    pangolin::Var<int> biwin("ui.size",10, 1, 20);
    pangolin::Var<float> bigs("ui.gs",10, 1E-3, 5);
    pangolin::Var<float> bigr("ui.gr",700, 1E-3, 200);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        if(video.Capture(images)) {
            for(int ni=0; ni < images.size(); ni++) {
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
                Gpu::DepthToVbo(imgv, imgf, sensor.fu, sensor.fv, sensor.u0, sensor.v0, 1.0f/1000.0f );
                Gpu::NormalsFromVbo(imgn, imgv);

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

        // Render stuff
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
        pangolin::FinishGlutFrame();
    }
}

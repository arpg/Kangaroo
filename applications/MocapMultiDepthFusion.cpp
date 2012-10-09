#include "common/ViconTracker.h"
#include "common/BaseDisplayCuda.h"
#include "common/RpgCameraOpen.h"

#include <boost/ptr_container/ptr_map.hpp>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include <kangaroo/kangaroo.h>

#include <SceneGraph/SceneGraph.h>

using namespace std;

const std::string ViconIp = "192.168.10.1";
const int MaxWidth = 640;
const int MaxHeight = 480;

struct Sensor {
    Sensor(std::string name, std::string vicon_ip, int w, int h)
        :name(name), tracker(name, vicon_ip), w(w), h(h),
         vbo(pangolin::GlArrayBuffer, w,h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW )
    {

    }

    std::string name;
    ViconTracking tracker;
    int w, h;
    pangolin::GlBufferCudaPtr vbo;
    Sophus::SE3 T_ws;
    float fu,fv,u0,v0;
};

struct GLSensor : SceneGraph::GLObject {
    GLSensor(Sensor& sensor)
        : GLObject(sensor.name), sensor(sensor)
    {
    }

    void DrawObjectAndChildren(SceneGraph::RenderMode renderMode)
    {
        // Update pose based on vicon
        SetPose(sensor.tracker.T_wf().matrix());

        // Render as usual
        GLObject::DrawObjectAndChildren(renderMode);
    }

    void DrawCanonicalObject()
    {
        // Just draw axis for time being
        SceneGraph::GLAxis::DrawUnitAxis();
    }

protected:
    Sensor& sensor;
};

typedef boost::ptr_map<std::string,Sensor> SensorPtrMap;

int main( int argc, char* argv[] )
{
    pangolin::View& container = SetupPangoGLWithCuda(1024, 768);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

    SceneGraph::GLSceneGraph graph;
    SceneGraph::GLGrid glGrid(10,1,true);
    SceneGraph::GLAxis glAxis;
    graph.AddChild(&glGrid);
    graph.AddChild(&glAxis);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAt(0,5,5,0,0,0,0,0,1)
    );

    SetupContainer(container, 1, 640.0f/480.0f);
    container[0].SetDrawFunction(SceneGraph::ActivateDrawFunctor(graph, s_cam))
                .SetHandler( new pangolin::Handler3D(s_cam, pangolin::AxisZ) );

    SensorPtrMap sensors;

    CameraDevice video = OpenRpgCamera(argc,argv,1);
    std::vector<rpg::ImageWrapper> images;

    Gpu::Image<unsigned short, Gpu::TargetDevice, Gpu::Manage> imgd(MaxWidth,MaxHeight);

    for(unsigned long frame=0; !pangolin::ShouldQuit(); ++frame)
    {
        if(video.Capture(images)) {
            // Which camera did this come from
            std::string sensor_name = images[0].Map.GetProperty("DeviceName");

            SensorPtrMap::iterator it = sensors.find(sensor_name);
            if( it == sensors.end() ) {
                // create sensor
                cout << "New sensor - " << sensor_name << endl;

                // Size of depthmap that this sensor will produce
                const int dw = images[0].Image.cols;
                const int dh = images[0].Image.rows;

                // Create and Insert into map of active sensors
                Sensor* newsensor = new Sensor(sensor_name, ViconIp, dw,dh);
                it = sensors.insert(sensor_name, newsensor ).first;

                // add to posegraph
                graph.AddChild(new GLSensor(*newsensor));
            }

            Sensor& sensor = *it->second;
            cout << "Data from " << sensor.name << endl;

            // Update depth map. (lets assume we have a kinect for now)
//            imgd.CopyFrom(images[0]);
        }

        // Render stuff
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
        pangolin::FinishGlutFrame();
    }
}

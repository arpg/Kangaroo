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
         vbo(pangolin::GlArrayBuffer, w,h,GL_FLOAT,4, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW )
    {
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
        T_vs = Sophus::SE3(SceneGraph::GLCart2T(x,y,z,p,q,r));
        cout << SceneGraph::GLT2Cart(T_vs.matrix()) << endl;
    }

    std::string name;
    ViconTracking tracker;
    int w, h;
    pangolin::GlBufferCudaPtr vbo;
    Sophus::SE3 T_ws;
    Sophus::SE3 T_vs;
    float fu,fv,u0,v0;
};

struct GLSensor : public SceneGraph::GLObject {
    GLSensor(Sensor& sensor)
        : GLObject(sensor.name), sensor(sensor)
    {
    }

    void DrawCanonicalObject()
    {
        glPushMatrix();
        glMultMatrixd(sensor.T_ws.matrix().data());

        // Just draw axis for time being
        SceneGraph::GLAxis::DrawAxis();

        glColor3f(1,1,1);
        pangolin::RenderVbo(sensor.vbo,sensor.w, sensor.h);

        glPopMatrix();
    }

protected:
    Sensor& sensor;
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
                .SetHandler( new pangolin::Handler3D(s_cam/*, pangolin::AxisZ*/) );

    SensorPtrMap sensors;

    CameraDevice video = OpenRpgCamera(argc,argv,1);
    std::vector<rpg::ImageWrapper> images;

    Gpu::Image<unsigned short, Gpu::TargetDevice, Gpu::Manage> imgd(MaxWidth,MaxHeight);
    Gpu::Image<float4, Gpu::TargetDevice, Gpu::Manage> imgv(MaxWidth,MaxHeight);

    ViconConnection vicon(ViconIp);

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
                    graph.AddChild(new GLSensor(*newsensor));
                }

                Sensor& sensor = *it->second;
    //            cout << "Data from " << sensor.name << endl;

                sensor.T_ws = sensor.tracker.T_wf() * sensor.T_vs;

                // Update depth map. (lets assume we have a kinect for now)
                imgd.CopyFrom<Gpu::TargetHost,Gpu::DontManage>(images[ni].Image);
                const float asusfocal = 570.342;
                const float df = asusfocal;

                // Convert to Vertex Buffer
                Gpu::DepthToVbo(imgv, imgd, df, df, dw/2.0f, dh/2.0f, 1.0f/1000.0f );

                // Copy to sensors vbo for display
                {
                    pangolin::CudaScopedMappedPtr var(sensor.vbo);
                    Gpu::Image<float4> dVbo((float4*)*var,dw,dh);
                    dVbo.CopyFrom(imgv);
                }
            }
        }

        // Render stuff
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
        pangolin::FinishGlutFrame();
    }
}

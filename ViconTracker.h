#include <vrpn_Tracker.h>
#include <quat.h>
#include <sophus/se3.h>
#include <Eigen/Geometry>
#include <boost/thread.hpp>

struct ViconTracking
{
    ViconTracking( std::string objectName, std::string host)
    {
        const std::string uri = objectName + "@" + host;
        m_object = new vrpn_Tracker_Remote( uri.c_str() );
        m_object->shutup = true;
        m_object->register_change_handler(this, &ViconTracking::c_callback );
        Start();
    }

    ~ViconTracking()
    {
        Stop();
        delete m_object;
    }

    void EventLoop() {
        while(m_run) {
            m_object->mainloop();
        }
    }

    void Start() {
        m_run = true;
        m_event_thread = boost::thread(&ViconTracking::EventLoop, this);
    }

    void Stop() {
        m_run = false;
        m_event_thread.join();
    }

    void TrackingEvent(const vrpn_TRACKERCB tData )
    {
        T_wf = Sophus::SE3( Sophus::SO3(Eigen::Quaterniond(tData.quat)),
            Eigen::Vector3d(tData.pos[0], tData.pos[1], tData.pos[2])
        );
    }

    static void VRPN_CALLBACK c_callback(void* userData, const vrpn_TRACKERCB tData )
    {
        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
        self->TrackingEvent(tData);
    }

    Sophus::SE3 T_wf;

    bool m_run;
    vrpn_Tracker_Remote* m_object;
    boost::thread m_event_thread;
};

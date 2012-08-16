#pragma once

#include <vrpn_Tracker.h>
#include <quat.h>
#include <sophus/se3.h>
#include <Eigen/Geometry>
#include <boost/thread.hpp>

class ViconTracking
{
public:
    ViconTracking( std::string objectName, std::string host)
        : m_connected(false), m_newdata(false), m_recordHistory(false)
    {
        const std::string uri = objectName + "@" + host;
        m_object = new vrpn_Tracker_Remote( uri.c_str() );
        m_object->shutup = true;
        m_object->register_change_handler(this, &ViconTracking::pose_callback );
//        m_object->register_change_handler(this, &ViconTracking::velocity_callback );
//        m_object->register_change_handler(this, &ViconTracking::workspace_callback );
        Start();
    }

    ~ViconTracking()
    {
        Stop();
        delete m_object;
    }

    inline const Sophus::SE3& T_wf()
    {
        m_newdata = false;
        return m_T_wf;
    }

    inline bool IsConnected()
    {
        return m_connected;
    }

    inline bool IsNewData()
    {
        return m_newdata;
    }

    void EventLoop() {
        while(m_run) {
            m_object->mainloop();
        }
    }

    inline void Start() {
        m_run = true;
        m_event_thread = boost::thread(&ViconTracking::EventLoop, this);
    }

    inline void Stop() {
        m_run = false;
        m_event_thread.join();
    }

    inline void RecordHistory(bool record = true)
    {
        m_recordHistory = record;
    }

    inline void ToggleRecordHistory()
    {
        m_recordHistory = !m_recordHistory;
    }

    inline void ClearHistory()
    {
        m_vecT_wf.clear();
    }

    inline const std::vector<Sophus::SE3>& History()
    {
        return m_vecT_wf;
    }

    inline void TrackingEvent(const vrpn_TRACKERCB tData )
    {
        m_T_wf = Sophus::SE3( Sophus::SO3(Eigen::Quaterniond(tData.quat)),
            Eigen::Vector3d(tData.pos[0], tData.pos[1], tData.pos[2])
        );
        m_connected = true;
        m_newdata = true;
        if(m_recordHistory) {
            m_vecT_wf.push_back(m_T_wf);
        }
    }

    static void VRPN_CALLBACK pose_callback(void* userData, const vrpn_TRACKERCB tData )
    {
        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
        self->TrackingEvent(tData);
    }

//    inline void TrackingEvent(const vrpn_TRACKERVELCB tData )
//    {
//        std::cout << "Velocity event" << std::endl;
//    }

//    static void VRPN_CALLBACK velocity_callback(void* userData, const vrpn_TRACKERVELCB tData )
//    {
//        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
//        self->TrackingEvent(tData);
//    }

//    inline void TrackingEvent(const vrpn_TRACKERWORKSPACECB tData )
//    {
//        std::cout << "Workspace event" << std::endl;
//    }

//    static void VRPN_CALLBACK workspace_callback(void* userData, const vrpn_TRACKERWORKSPACECB tData )
//    {
//        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
//        self->TrackingEvent(tData);
//    }


protected:
    Sophus::SE3 m_T_wf;
    std::vector<Sophus::SE3> m_vecT_wf;

    bool m_connected;
    bool m_newdata;
    bool m_run;
    bool m_recordHistory;

    vrpn_Tracker_Remote* m_object;
    boost::thread m_event_thread;
};

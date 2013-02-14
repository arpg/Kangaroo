#pragma once

#include "Tracking.h"

#include <vrpn_Tracker.h>
#include <quat.h>

#include <boost/thread.hpp>


class ViconConnection
{
public:
    inline ViconConnection(const std::string& host)
        : m_run(true), m_host(host)
    {
        m_connection = vrpn_get_connection_by_name(host.c_str());
        m_thread = boost::thread(&ViconConnection::EventLoop, this);
    }

    inline ~ViconConnection() {
        m_run = false;
        m_thread.join();
        m_connection->removeReference();
    }

    inline void EventLoop() {
        while(m_run) {
            m_connection->mainloop();
        }
    }

    inline std::string HostName() {
        return m_host;
    }

    inline vrpn_Connection* Connection() {
        return m_connection;
    }

protected:
    bool m_run;
    vrpn_Connection* m_connection;
    std::string m_host;
    boost::thread m_thread;
};

class ViconTracking
    : public Tracking
{
public:
    ViconTracking( std::string objectName, std::string host)
    {
        const std::string uri = objectName + "@" + host;
        m_object = new vrpn_Tracker_Remote( uri.c_str() );
        RegisterHandlers();
        StartThread();
    }

    ViconTracking( std::string objectName, ViconConnection& sharedConnection)
    {
        const std::string uri = objectName + "@" + sharedConnection.HostName();
        m_object = new vrpn_Tracker_Remote( uri.c_str(), sharedConnection.Connection() );
        RegisterHandlers();
    }

    ~ViconTracking()
    {
        StopThread();
        UnregisterHandlers();
        delete m_object;
    }
    
protected:
    
    inline void RegisterHandlers() {
        m_object->shutup = true;
        m_object->register_change_handler(this, &ViconTracking::pose_callback );
        m_object->register_change_handler(this, &ViconTracking::velocity_callback );
        m_object->register_change_handler(this, &ViconTracking::workspace_callback );
        m_object->request_t2r_xform();
        m_object->request_u2s_xform();
        m_object->request_workspace();
    }

    inline void UnregisterHandlers() {
        m_object->unregister_change_handler(this, &ViconTracking::pose_callback );
        m_object->register_change_handler(this, &ViconTracking::velocity_callback );
        m_object->register_change_handler(this, &ViconTracking::workspace_callback );
    }    

    static void VRPN_CALLBACK pose_callback(void* userData, const vrpn_TRACKERCB tData )
    {
        struct timeval tv;
        gettimeofday(&tv, 0);
        double time_now_s = tv.tv_sec + 1e-6 * (tv.tv_usec);
        
        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
        self->TrackingEvent(
            Sophus::SE3d( Sophus::SO3d(Eigen::Quaterniond(tData.quat)), Eigen::Vector3d(tData.pos[0], tData.pos[1], tData.pos[2] ) ),
            time_now_s,
            tData.msg_time.tv_sec + 1e-6 * tv.tv_usec
        );
    }

    static void VRPN_CALLBACK velocity_callback(void* userData, const vrpn_TRACKERVELCB tData )
    {
        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
        std::cout << "Velocity event!" << std::endl;        
    }

    static void VRPN_CALLBACK workspace_callback(void* userData, const vrpn_TRACKERWORKSPACECB tData )
    {
        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
        std::cout << "Workspace event!" << std::endl;        
    }

    void EventLoop() {
        while(m_run) {
            m_object->mainloop();
        }
    }

    // This should only be started if an external ViconConnection object isn't
    // being used - (If only one tracker is running)
    inline void StartThread() {
        m_run = true;
        m_event_thread = boost::thread(&ViconTracking::EventLoop, this);
    }

    // Stop thread if it exists
    inline void StopThread() {
        m_run = false;
        m_event_thread.join();
    }

    vrpn_Tracker_Remote* m_object;
    boost::thread m_event_thread;
};

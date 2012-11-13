#pragma once

#include <vrpn_Tracker.h>
#include <quat.h>
#include <sophus/se3.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
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
{
public:
    ViconTracking( std::string objectName, std::string host)
        : m_connected(false), m_newdata(false), m_run(false),
          m_recordHistory(false)
    {
        const std::string uri = objectName + "@" + host;
        m_object = new vrpn_Tracker_Remote( uri.c_str() );
        WorkspaceReset();
        RegisterHandlers();
        StartThread();
    }

    ViconTracking( std::string objectName, ViconConnection& sharedConnection)
        : m_connected(false), m_newdata(false), m_recordHistory(false)
    {
        const std::string uri = objectName + "@" + sharedConnection.HostName();
        m_object = new vrpn_Tracker_Remote( uri.c_str(), sharedConnection.Connection() );
        RegisterHandlers();
    }

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

    ~ViconTracking()
    {
        StopThread();
        UnregisterHandlers();
        delete m_object;
    }

    inline void WorkspaceReset() {
        m_workspace_min = Eigen::Vector3d(+1E6,+1E6,+1E6);
        m_workspace_max = Eigen::Vector3d(-1E6,-1E6,-1E6);
    }

    inline const Eigen::Vector3d& WorkspaceMin() {
        return m_workspace_min;
    }

    inline const Eigen::Vector3d& WorkspaceMax() {
        return m_workspace_max;
    }

    inline const Eigen::Vector3d WorkspaceSize() {
        return m_workspace_max - m_workspace_min;
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

        m_workspace_min = ElementwiseMin(m_workspace_min, m_T_wf.translation());
        m_workspace_max = ElementwiseMax(m_workspace_max, m_T_wf.translation());

        if(m_recordHistory) {
            m_vecT_wf.push_back(m_T_wf);
        }
    }

    inline void TrackingEvent(const vrpn_TRACKERVELCB tData )
    {
        std::cout << "Velocity event" << std::endl;
    }

    inline void TrackingEvent(const vrpn_TRACKERWORKSPACECB tData )
    {
        m_workspace_min = Eigen::Map<const Eigen::Vector3d>(tData.workspace_min);
        m_workspace_max = Eigen::Map<const Eigen::Vector3d>(tData.workspace_max);
    }

    static void VRPN_CALLBACK pose_callback(void* userData, const vrpn_TRACKERCB tData )
    {
        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
        self->TrackingEvent(tData);
    }

    static void VRPN_CALLBACK velocity_callback(void* userData, const vrpn_TRACKERVELCB tData )
    {
        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
        self->TrackingEvent(tData);
    }

    static void VRPN_CALLBACK workspace_callback(void* userData, const vrpn_TRACKERWORKSPACECB tData )
    {
        ViconTracking* self = reinterpret_cast<ViconTracking*>(userData);
        self->TrackingEvent(tData);
    }

protected:

    inline Eigen::Vector3d ElementwiseMin(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
    {
        return Eigen::Vector3d(std::min(a(0), b(0)), std::min(a(1), b(1)), std::min(a(2), b(2)) );
    }

    inline Eigen::Vector3d ElementwiseMax(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
    {
        return Eigen::Vector3d(std::max(a(0), b(0)), std::max(a(1), b(1)), std::max(a(2), b(2)) );
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

    Eigen::Vector3d m_workspace_min;
    Eigen::Vector3d m_workspace_max;

    Sophus::SE3 m_T_wf;
    std::vector<Sophus::SE3> m_vecT_wf;

    bool m_connected;
    bool m_newdata;
    bool m_run;
    bool m_recordHistory;

    vrpn_Tracker_Remote* m_object;
    boost::thread m_event_thread;
};

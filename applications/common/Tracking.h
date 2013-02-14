#pragma once

#include <boost/function.hpp>
#include <sophus/se3.h>

typedef boost::function<void (const Sophus::SE3d& T_wf, double system_time_s, double dev_time_s)>
    TrackingDataCallback;

class Tracking
{
public:
    Tracking()
        : m_connected(false), m_newdata(false), m_run(false),
          m_recordHistory(false)
    {
        WorkspaceReset();        
    }
    
    inline const Sophus::SE3d& T_wf()
    {
        m_newdata = false;
        return m_T_wf;
    }
    
    inline void RegisterTrackingCallback(const TrackingDataCallback& callback)
    {
        m_TrackingCallback = callback;
    }

    inline void WorkspaceReset() {
        m_workspace_min = Eigen::Vector3d(+1E6,+1E6,+1E6);
        m_workspace_max = Eigen::Vector3d(-1E6,-1E6,-1E6);
    }

    inline Eigen::Vector3d& WorkspaceMin() {
        return m_workspace_min;
    }

    inline Eigen::Vector3d& WorkspaceMax() {
        return m_workspace_max;
    }

    inline const Eigen::Vector3d WorkspaceSize() {
        return m_workspace_max - m_workspace_min;
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

    inline void SetOffset(const Sophus::SE3d& T_offset)
    {
        m_T_offset = T_offset;
    }

    inline const std::vector<Sophus::SE3d>& History()
    {
        return m_vecT_wf;
    }
    
protected:
    inline void TrackingEvent(const Sophus::SE3d& T_wf, double system_time_s, double dev_time_s)
    {
        m_T_wf = m_T_offset * T_wf;
        m_connected = true;
        m_newdata = true;

        m_workspace_min = ElementwiseMin(m_workspace_min, m_T_wf.translation());
        m_workspace_max = ElementwiseMax(m_workspace_max, m_T_wf.translation());

        if(m_recordHistory) {
            m_vecT_wf.push_back(m_T_wf);
        }
        
        if(!m_TrackingCallback.empty()) {
            m_TrackingCallback(m_T_wf, system_time_s, dev_time_s);
        }
    }
    
    inline Eigen::Vector3d ElementwiseMin(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
    {
        return Eigen::Vector3d(std::min(a(0), b(0)), std::min(a(1), b(1)), std::min(a(2), b(2)) );
    }

    inline Eigen::Vector3d ElementwiseMax(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
    {
        return Eigen::Vector3d(std::max(a(0), b(0)), std::max(a(1), b(1)), std::max(a(2), b(2)) );
    }
    
    Eigen::Vector3d m_workspace_min;
    Eigen::Vector3d m_workspace_max;

    Sophus::SE3d m_T_offset;
    Sophus::SE3d m_T_wf;    
    std::vector<Sophus::SE3d> m_vecT_wf;
    
    TrackingDataCallback   m_TrackingCallback;    
    
    bool m_connected;
    bool m_newdata;
    bool m_run;
    bool m_recordHistory;
};


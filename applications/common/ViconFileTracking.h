#pragma once

#include "Tracking.h"
#include <RPG/Devices/VirtualDevice.h>

#include <boost/thread.hpp>

class ViconFileTracking
    : public Tracking
{
public:
    ViconFileTracking( std::string objectName, std::string baseDir)
        : index(0)
    {
        ReadViconFile(baseDir + "/vicon.txt");
        StartThread();
    }

    ~ViconFileTracking()
    {
        StopThread();
    }
    
protected:
    bool ReadViconFile(std::string filename)
    {
        double time_sys;
        double time_vic;
        
        Sophus::Vector6d vec;
        std::ifstream fin;
    
        fin.open(filename.c_str());
    
        const int MAX_SPACES = 100000;
    
        if(fin.is_open()) {
            while(!fin.eof()) {
                fin >> time_sys;
                fin.ignore(MAX_SPACES,',');
                fin >> time_vic;
                for(int i=0; i<6; i++) {
                    fin.ignore(MAX_SPACES,',');
                    fin >> vec(i);
                }
                vecT_wp.push_back( Sophus::SE3(SceneGraph::GLCart2T(vec)) );
                vecSystemTime.push_back(time_sys);
                vecDeviceTime.push_back(time_vic);
            }
            fin.close();
            return true;
        }else {
            return false;
        }
    }
    
    void EventLoop() {        
        while(m_run && index < vecT_wp.size() ) {
            if(index==0) {
                VirtualDevice::PushTime(vecSystemTime[index]);
            }else{
                VirtualDevice::PopAndPushTime(vecSystemTime[index]);                
            }
            VirtualDevice::WaitForTime(vecSystemTime[index]);
            
            TrackingEvent(vecT_wp[index], vecSystemTime[index], vecDeviceTime[index] );
            ++index;
        }
    }

    inline void StartThread() {
        m_run = true;
        m_event_thread = boost::thread(&ViconFileTracking::EventLoop, this);
    }

    // Stop thread if it exists
    inline void StopThread() {
        m_run = false;
        m_event_thread.interrupt();
        m_event_thread.join();
    }

    size_t index;
    std::vector<Sophus::SE3d> vecT_wp;
    std::vector<double> vecSystemTime;
    std::vector<double> vecDeviceTime;
    
    boost::thread m_event_thread;
};

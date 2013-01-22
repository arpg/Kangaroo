#include <iostream>
#include <boost/thread.hpp>
#include <pangolin/pangolin.h>
#include <RPG/Devices/IMU/IMUDevice.h>

using namespace pangolin;
using namespace std;

#include <pangolin/display_internal.h>

struct Application
{
    Application()
        : log(1024*1024), plotter(&log)
    {
        // Setup IMU
        imu.SetProperty("GetAHRS", true);
        imu.SetProperty("GetGyro", true);
        imu.SetProperty("GetAccelerometer", true);
        imu.SetProperty("HzAHRS", 1000);

        if( imu.InitDriver( "MicroStrain" ) ) {
            IMUDriverDataCallback f = boost::bind(&Application::NewIMUData, this, _1);
            imu.RegisterIMUDataCallback( f );
        }else{
            std::cout << "Invalid input device." << std::endl;
            exit(-1);
        }

        // Create OpenGL window in single line thanks to GLUT
        pangolin::CreateGlutWindowAndBind("Main",640,480);
        glEnable (GL_BLEND);
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
        DisplayBase().AddDisplay(plotter);
    }

    ~Application()
    {
        imu.DeinitDriver();
    }


    void NewIMUData(IMUData data)
    {
        log.Log(
            data.gyro(0), data.gyro(1), data.gyro(2),
            data.accel(0), data.accel(1), data.accel(2)
        );
    }

    void Run()
    {
      // Default hooks for exiting (Esc) and fullscreen (tab).
      while( !pangolin::ShouldQuit() )
      {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        pangolin::FinishGlutFrame();
        boost::this_thread::sleep(boost::posix_time::milliseconds(1E3/60.0));
      }

    }

protected:
    IMUDevice imu;
    DataLog log;
    Plotter plotter;
};

int main( int /*argc*/, char* argv[] )
{
    Application app;
    app.Run();
    return 0;
}

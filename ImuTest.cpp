#include <boost/signals2/mutex.hpp>
#include <Eigen/Eigen>

#include <SimpleGui/Gui.h>
#include <SimpleGui/GetPot>
#include <SimpleGui/GLMesh.h>

#include <RPG/Devices/IMU/IMUDevice.h>
#include <Mvlpp/Mvl.h>

using namespace std;
using namespace Eigen;

const char* MESH_NAME = "./Models/beatle-no-wheels-no-interior-embedded-texture.blend";

class Application
{
public:
    Application()
        : window(0, 0, 1024, 768, __FILE__ ),
          meshtest()
    {
        Init();
    }

    void Init()
    {
    //    Cam.SetProperty("property", val);

        // Initialise driver
        if( imu.InitDriver( "MicroStrain" ) ) {
            IMUDriverDataCallback f = boost::bind(&Application::NewIMUData, this, _1);
            imu.RegisterDataCallback( f );
        }else{
            std::cout << "Invalid input device." << std::endl;
        }

        meshtest.Init(MESH_NAME);

//        window.AddChildToRoot(&meshtest);
        window.AddPreRenderCallback(Application::PreRender, this);
        window.AddPostRenderCallback(Application::PostRender, this);
    }

    void NewIMUData(const IMUData& data)
    {
//        updateMutex.lock();
        rotation = data.rotation;
//        updateMutex.unlock();
    }

    static void PreRender(GLWindow*, void* data)
    {

    }

    static void PostRender(GLWindow*, void* data)
    {
        Application* self = static_cast<Application*>(data);
        Eigen::Matrix4d T;
        T.setZero();
        T(3,3) = 1;
        self->updateMutex.lock();
        T.block<3,3>(0,0) = self->rotation.toRotationMatrix();
        self->updateMutex.unlock();

        glPushMatrix();
        glMultMatrixd(T.data());
        self->meshtest.draw();
        glPopMatrix();
    }

    int Run()
    {
        return window.Run();
    }

protected:
    boost::signals2::mutex updateMutex;
    Quaterniond rotation;
    IMUDevice imu;
    GLWindow window;
    GLMesh meshtest;
};

int main (int argc, char** argv){
    Application app;
    return app.Run();
}

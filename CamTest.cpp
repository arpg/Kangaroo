#include <SimpleGui/Gui.h>
#include <SimpleGui/GetPot>

#include <dc1394/dc1394.h>
#include <dc1394/conversions.h>


using namespace std;
using namespace Eigen;

const dc1394video_mode_t DEFAULT_MODE = DC1394_VIDEO_MODE_640x480_MONO8;
const unsigned DEFAULT_DMA = 8;
unsigned int m_nImageWidth = 640;
unsigned int m_nImageHeight = 480;
unsigned int m_nImgSize = m_nImageWidth * m_nImageHeight;


void SetAbsFrameRate(dc1394camera_t* camera, float val)
{
    dc1394error_t err = dc1394_feature_set_mode(camera, DC1394_FEATURE_FRAME_RATE, DC1394_FEATURE_MODE_MANUAL);
    if (err < 0) {
        std::cerr << "Could not set manual frame-rate" << std::endl;
    }

    err = dc1394_feature_set_absolute_control(camera, DC1394_FEATURE_FRAME_RATE, DC1394_ON);
    if (err < 0) {
        std::cerr << "Could not set absolute control for frame-rate" << std::endl;
    }

    err = dc1394_feature_set_absolute_value(camera, DC1394_FEATURE_FRAME_RATE, val);
    if (err < 0) {
        std::cerr << "Could not set frame-rate value" << std::endl;
    }
}

void SetAbsShutterTime(dc1394camera_t* camera, float val) {
    dc1394error_t err = dc1394_feature_set_mode(camera, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_MANUAL);
    if (err < 0) {
        std::cerr << "Could not set manual shutter mode" << std::endl;
    }

    err = dc1394_feature_set_absolute_control(camera, DC1394_FEATURE_SHUTTER, DC1394_ON);
    if (err < 0) {
        std::cerr << "Could not set absolute control for shutter" << std::endl;
    }

    err = dc1394_feature_set_absolute_value(camera, DC1394_FEATURE_SHUTTER, val);
    if (err < 0) {
        std::cerr << "Could not set shutter value" << std::endl;
    }
}

void SetAbsExposure(dc1394camera_t* camera, float val)
{
    dc1394error_t err = dc1394_feature_set_mode(camera, DC1394_FEATURE_EXPOSURE, DC1394_FEATURE_MODE_MANUAL);
    if (err < 0) {
        std::cerr << "Could not set manual shutter mode" << std::endl;
    }

    err = dc1394_feature_set_absolute_control(camera, DC1394_FEATURE_EXPOSURE, DC1394_ON);
    if (err < 0) {
        std::cerr << "Could not set absolute control for shutter" << std::endl;
    }

    err = dc1394_feature_set_absolute_value(camera, DC1394_FEATURE_EXPOSURE, val);
    if (err < 0) {
        std::cerr << "Could not set exposure value" << std::endl;
    }
}

bool EnableTrigger(dc1394camera_t* camera, int src)
{
    dc1394trigger_source_t source = dc1394trigger_source_t(DC1394_TRIGGER_SOURCE_0 + src);

    if(dc1394_external_trigger_set_power(camera, DC1394_ON)
       != DC1394_SUCCESS) {
        return false;
    }

    if(dc1394_external_trigger_set_source(camera, source)
       != DC1394_SUCCESS) {
        return false;
    }

    return true;
}

bool DisableTrigger(dc1394camera_t* camera)
{
    return dc1394_external_trigger_set_power(camera, DC1394_OFF) ==
            DC1394_SUCCESS;
}

bool SetTriggerMode(dc1394camera_t* camera, dc1394trigger_mode_t mode)
{
    if(dc1394_external_trigger_set_mode(camera, mode)
       != DC1394_SUCCESS) {
        return false;
    }

    return true;
}

bool SetTriggerPolarity(dc1394camera_t* camera, dc1394trigger_polarity_t polarity)
{
    dc1394error_t e = dc1394_external_trigger_set_polarity(camera, polarity);

    if(e != DC1394_SUCCESS) {
        return false;
    }

    return true;
}


void SendSoftwareTrigger(dc1394camera_t* camera)
{
    dc1394_software_trigger_set_power(camera, DC1394_ON);
}

dc1394error_t SetCameraProperties(
        dc1394camera_t* camera,
        dc1394video_mode_t mode,
        dc1394framerate_t framerate,
        unsigned dma_channels
        ) {
    dc1394error_t e;

    e = dc1394_video_set_operation_mode(camera, DC1394_OPERATION_MODE_1394B);
    DC1394_ERR_RTN(e,  "Could not set operation mode");

    e = dc1394_video_set_iso_speed(camera, DC1394_ISO_SPEED_400);
    DC1394_ERR_RTN(e,  "Could not set iso speed");

    e = dc1394_video_set_mode(camera, mode);
    DC1394_ERR_RTN(e,  "Could not set video mode");

    e = dc1394_video_set_framerate(camera, framerate);
    DC1394_ERR_RTN(e,  "Could not set framerate");

    e = dc1394_capture_setup(camera, dma_channels, DC1394_CAPTURE_FLAGS_DEFAULT);
    DC1394_ERR_RTN(e,  "Could not setup camera. Make sure that the video mode and framerate are supported by your camera.");

    return DC1394_SUCCESS;
}

class Application
{
public:
    Application()
        : window(0, 0, 1024, 768, __FILE__ )
    {
        Init();
    }

    int Init()
    {
        dc1394error_t e;

        // here we connect to the firefly and see if it's alive
        m_pBus = dc1394_new();

        dc1394camera_list_t* pCameraList = NULL;
        e = dc1394_camera_enumerate(m_pBus, &pCameraList);

        cout << "Found:" << endl;
        unsigned int CamNum = 0;
        for( int ii = 0; ii < (int) pCameraList->num; ii++ ){
            dc1394camera_t* pCam;
            pCam = dc1394_camera_new(m_pBus, pCameraList->ids[ii].guid);
            printf("Model %s\n", pCam->model);

            // the model
            std::string CamModel(pCam->model);
            if( CamModel.find("Firefly") != std::string::npos ){
                m_pCam[CamNum++] = pCam;
            } else {
                // close camera
            }
        }

        if (pCameraList->num < 2) {
            cerr << "Didn't find enough cameras" << endl;
            exit(-1);
        }

        // free the camera list
        dc1394_camera_free_list(pCameraList);
        cout << "Using camera with GUID " << m_pCam[0]->guid << endl;
        cout << "Using camera with GUID " << m_pCam[1]->guid << endl;

        // Get width / height for mode
        //    dc1394_get_image_size_from_video_mode(m_pCam[0], DEFAULT_MODE, &m_nImageWidth, &m_nImageHeight);
        //    m_nImgSize = m_nImageWidth * m_nImageHeight;

        // Get highest framerate for mode
        dc1394framerates_t vFramerates;
        e = dc1394_video_get_supported_framerates(m_pCam[0], DEFAULT_MODE, &vFramerates);
        DC1394_ERR_RTN(e, "Could not get framerates");
        dc1394framerate_t nFramerate = vFramerates.framerates[vFramerates.num - 1];

        // Set up properties for cameras
        SetCameraProperties(m_pCam[0], DEFAULT_MODE, nFramerate, DEFAULT_DMA );
        SetCameraProperties(m_pCam[1], DEFAULT_MODE, nFramerate, DEFAULT_DMA );

        // Set framerate feature. (This is distinct from trigger rate)
        SetAbsFrameRate(m_pCam[0], 60);
        SetAbsFrameRate(m_pCam[1], 60);

        //    // Force fixed shutter time
        //    SetAbsShutterTime(m_pCam[0], 1.0 / 120.0);
        //    SetAbsShutterTime(m_pCam[1], 1.0 / 120.0);

        //    // print camera features
        //    dc1394featureset_t vFeatures;
        //    e = dc1394_feature_get_all(m_pCam[0], &vFeatures);
        //    if (e != DC1394_SUCCESS) {
        //        dc1394_log_warning("Could not get feature set");
        //    } else {
        //        dc1394_feature_print_all( &vFeatures, stdout );
        //    }
        //    e = dc1394_feature_get_all(m_pCam[1], &vFeatures);
        //    if (e != DC1394_SUCCESS) {
        //        dc1394_log_warning("Could not get feature set");
        //    } else {
        //        dc1394_feature_print_all( &vFeatures, stdout );
        //    }

        // initiate transmission
        e = dc1394_video_set_transmission(m_pCam[0], DC1394_ON);
        DC1394_ERR_RTN(e, "Could not start camera iso transmission");
        e = dc1394_video_set_transmission(m_pCam[1], DC1394_ON);
        DC1394_ERR_RTN(e, "Could not start camera iso transmission");

        window.AddPostRenderCallback( Application::PostRender, this);
    }

    int Run()
    {
        return window.Run();
    }

    void GlSanityTest()
    {
        glDisable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glClearColor (0.0, 0.0, 0.0, 0.0);
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glColor3f (1.0, 1.0, 1.0);
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
        glBegin(GL_POLYGON);
        glVertex3f (0.25, 0.25, 0.0);
        glVertex3f (0.75, 0.25, 0.0);
        glVertex3f (0.75, 0.75, 0.0);
        glVertex3f (0.25, 0.75, 0.0);
        glEnd();
    }

    void CaptureDraw()
    {
        //  capture frame
        dc1394error_t e;
        dc1394video_frame_t * pFrame[2];
        dc1394capture_policy_t nPolicy = DC1394_CAPTURE_POLICY_WAIT;

        e = dc1394_capture_dequeue(m_pCam[0], nPolicy, &pFrame[0]);
        if (e != DC1394_SUCCESS)
            std::cerr << "Warning: lost a 0 frame!" << std::endl;

        if(pFrame[0]) {
            //memcpy(fbp1,pFrame[0]->image, m_nImgSize);
            e = dc1394_capture_enqueue(m_pCam[0], pFrame[0]);
        }

        e = dc1394_capture_dequeue(m_pCam[1], nPolicy, &pFrame[1]);
        if (e != DC1394_SUCCESS)
            std::cerr << "Warning: lost a 1 frame!" << std::endl;

        if(pFrame[1]) {
            //memcpy(fbp2,pFrame[1]->image, m_nImgSize);
            e = dc1394_capture_enqueue(m_pCam[1], pFrame[1]);
        }

        if(pFrame[0] && pFrame[1]) {
            nFrames++;
        }
    }

    static void PostRender(GLWindow*, void* data)
    {
        Application* self = (Application*)data;
        self->CaptureDraw();
    }

    GLWindow window;
    dc1394_t* m_pBus;
    dc1394camera_t* m_pCam[2];
    int nFrames;
};

int main (int argc, char** argv){
    Application app;
    return app.Run();
}

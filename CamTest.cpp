#include <SimpleGui/Gui.h>
#include <SimpleGui/GetPot>

#include <dc1394/dc1394.h>
#include <dc1394/conversions.h>


using namespace std;
using namespace Eigen;

const dc1394video_mode_t DEFAULT_MODE = DC1394_VIDEO_MODE_640x480_MONO8;
const unsigned DEFAULT_DMA = 8;

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

bool EnableTrigger(dc1394camera_t* camera, dc1394trigger_source_t source)
{
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

string convBase(unsigned long v, long base)
{
    string digits = "0123456789abcdef";
    string result;
    if((base < 2) || (base > 16)) {
        result = "Error: base out of range.";
    }
    else {
        do {
            result = digits[v % base] + result;
            v /= base;
        }
        while(v);
    }
    return result;
}

// Address relative to CONFIG_ROM_BASE = 0xFFFFFF000000
//#define TRIGGER_MODE    0xF00830
#define PIO_DIRECTION   0xF011F8
#define STROBE_CTRL_INQ 0xF01300
#define STROBE_0_INQ    0xF01400
#define STROBE_0_CNT    0xF01500

inline uint32_t PointGreyBitmask(unsigned bitIndex)
{
    // Count bitIndex from MSB
    return 0x80000000 >> bitIndex;
}

inline void SetRegisterMasked(dc1394camera_t* cam, uint64_t offset, uint32_t mask, uint32_t value)
{
    dc1394error_t e;

    cout << convBase(offset,16) << " ";
    uint32_t val;
    e = dc1394_get_register(cam, offset, &val);
    assert(e == DC1394_SUCCESS);

    cout << "was: " << convBase(val,2);
    val |= value & mask;
    val &= value | ~mask;
    e = dc1394_set_register(cam, offset, val);
    assert(e == DC1394_SUCCESS);
    cout << ", set: " << convBase(val,2);

    e = dc1394_get_register(cam, offset, &val);
    assert(e == DC1394_SUCCESS);
    cout << ", is: " << convBase(val,2) << endl;
}

void SetPointGreyGPIO(dc1394camera_t* cam, unsigned gpioPin, bool useAsOutput )
{
    const uint32_t pinVal = useAsOutput ? 1 : 0;
    const uint32_t RegVal = pinVal * PointGreyBitmask(gpioPin);
    // Set bit in register corresponding to gpioPin only.
    SetRegisterMasked(cam, PIO_DIRECTION, RegVal, RegVal );
}

int SetPointGreyStobeWithShutter(dc1394camera_t* cam, uint32_t gpioPin)
{
    dc1394error_t e;
    uint32_t strobe_inq;
    e = dc1394_get_register(cam, STROBE_CTRL_INQ, &strobe_inq);
    assert(e == DC1394_SUCCESS);
    cout << "Strobe presence inq: " << convBase(strobe_inq,16) << endl;

    e = dc1394_get_register(cam, STROBE_0_INQ + sizeof(uint32_t)*gpioPin, &strobe_inq);
    assert(e == DC1394_SUCCESS);
    cout << "Strobe 1 inq: " << convBase(strobe_inq,16) << endl;


    SetPointGreyGPIO(cam, gpioPin, true);

    uint32_t strobe_cnt_offset = STROBE_0_CNT + sizeof(uint32_t)*gpioPin;
    uint32_t regVal = 0;
    regVal |= PointGreyBitmask(6); // Enable strobe
    regVal |= PointGreyBitmask(7); // High active output
    // bits 8-19 are delay value
    // bits 20-31 are signal duration (0 corresponds to exposure time)
    SetRegisterMasked(cam, strobe_cnt_offset, 0xFFFFFFFF, regVal);
}

void SetTriggerCam1FromCam2(dc1394camera_t* cam1, dc1394camera_t* cam2)
{
    // Setup cameras symetrically, so it doesn't matter in which order they're enumerated
    // GPIO1 of cam2 connected to GPIO0 of cam1
    // GPIO1 of cam1 connected to GPIO0 of cam2

    // Cam1 has external trigger on GPIO0
    SetPointGreyGPIO(cam1, 0, false);
    SetPointGreyGPIO(cam2, 0, false);
    EnableTrigger(cam1, DC1394_TRIGGER_SOURCE_0);
    SetTriggerMode(cam1, DC1394_TRIGGER_MODE_0);

    // Cam2 has strobe on GPIO1
    SetPointGreyStobeWithShutter(cam2, 1);
}

int CheckGLErrors()
{
  int errCount = 0;
  for(GLenum currError = glGetError(); currError != GL_NO_ERROR; currError = glGetError())
  {
      cerr << "ERROR!!" << endl;
    //Do something with `currError`.
    ++errCount;
  }

  return errCount;
}

void glDrawTexturesQuad(float t, float b, float l, float r)
{
    glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2f(l,b);
    glTexCoord2f(1,0); glVertex2f(r,b);
    glTexCoord2f(1,1); glVertex2f(r,t);
    glTexCoord2f(0,1); glVertex2f(l,t);
    glEnd();
}

class Application
{
public:
    Application()
        : window(0, 0, 640*2, 480, __FILE__ )
    {
        Init();
    }

    int Run()
    {
//        return window.Run();
        while(1)
        {
            CaptureDraw();
            window.swap_buffers();
        }
    }

    static void PostRender(GLWindow*, void* data)
    {
        Application* self = (Application*)data;
        self->CaptureDraw();
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
        uint32_t width, height;
        dc1394_get_image_size_from_video_mode(m_pCam[0], DEFAULT_MODE, &width, &height);
        m_width = width;
        m_height = height;
        m_sizeBytes = m_width * m_height;

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

        // Setup Triggering
        SetTriggerCam1FromCam2(m_pCam[1], m_pCam[0]);

        // initiate transmission
        e = dc1394_video_set_transmission(m_pCam[0], DC1394_ON);
        DC1394_ERR_RTN(e, "Could not start camera iso transmission");
        e = dc1394_video_set_transmission(m_pCam[1], DC1394_ON);
        DC1394_ERR_RTN(e, "Could not start camera iso transmission");

        // Create two OpenGL textures for stereo images
        glGenTextures(2, m_glTex);

        // Allocate texture memory on GPU
        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,0);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

        window.AddPostRenderCallback( Application::PostRender, this);
    }

    void CaptureDraw()
    {
        // Capture frame
        dc1394error_t e;
        dc1394video_frame_t * pFrame[2];
        dc1394capture_policy_t nPolicy = DC1394_CAPTURE_POLICY_WAIT;

        dc1394_capture_dequeue(m_pCam[0], nPolicy, &pFrame[0]);

        if(pFrame[0] ) {
            if(!dc1394_capture_is_frame_corrupt(m_pCam[0],pFrame[0]))
            {
                glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
                glTexSubImage2D(GL_TEXTURE_2D,0,0,0,m_width,m_height,GL_LUMINANCE,GL_UNSIGNED_BYTE,pFrame[0]->image);
            }else{
                cerr << "Corrupt frame" << endl;
            }
            e = dc1394_capture_enqueue(m_pCam[0], pFrame[0]);
        }else{
            cerr << "No frame" << endl;
        }

        dc1394_capture_dequeue(m_pCam[1], nPolicy, &pFrame[1]);

        if(pFrame[1]) {
            if( !dc1394_capture_is_frame_corrupt(m_pCam[1],pFrame[1]) ) {
                glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
                glTexSubImage2D(GL_TEXTURE_2D,0,0,0,m_width,m_height,GL_LUMINANCE,GL_UNSIGNED_BYTE,pFrame[1]->image);
            }else{
                cerr << "Corrupt frame" << endl;
            }
            e = dc1394_capture_enqueue(m_pCam[1], pFrame[1]);
        }else{
            cerr << "Bad frame" << endl;
        }

        if(pFrame[0] && pFrame[1]) {
            nFrames++;
        }

        // Display frames
        glDisable(GL_LIGHTING);
        glDisable(GL_COLOR_MATERIAL );

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glClearColor (0.0, 0.0, 0.0, 0.0);
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_TEXTURE_2D);

        glColor3f (1.0, 1.0, 1.0);

        glBindTexture(GL_TEXTURE_2D, m_glTex[0]);
        glDrawTexturesQuad(1,-1,1,0);

        glBindTexture(GL_TEXTURE_2D, m_glTex[1]);
        glDrawTexturesQuad(-1,1,-1,0);

        glDisable(GL_TEXTURE_2D);
    }

    GLWindow window;
    dc1394_t* m_pBus;
    dc1394camera_t* m_pCam[2];
    GLuint m_glTex[2];

    GLsizei m_width;
    GLsizei m_height;
    size_t m_sizeBytes;

    int nFrames;
};

int main (int argc, char** argv){
    Application app;
    return app.Run();
}

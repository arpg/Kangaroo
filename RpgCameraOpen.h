#include <pangolin/pangolin.h>
#include <RPG/Devices/Camera/CameraDevice.h>

//! Convenience method to load camera based on string URI containing PropertyMap and device.
CameraDevice OpenRpgCamera( const std::string& str_uri);

//! Convenience method to retrieve pangolin video wrapped as RPG Video object.
CameraDevice OpenPangoCamera( const std::string& str_uri, const std::string& str_uri2 = "");

//////////////////////////////////////////////////////////////////////

class PangolinRpgVideoAdapterDriver : public CameraDriver
{
public:
    static const int MAX_CAMS = 2;

    bool Capture( std::vector<rpg::ImageWrapper>& vImages );
    bool Init();

protected:
    size_t m_numCams;
    pangolin::VideoInput pangovid[MAX_CAMS];
};

inline CameraDevice OpenRpgCamera( const std::string& str_uri)
{
    CameraDevice camera;
    pangolin::Uri uri = pangolin::ParseUri(str_uri);

    for(std::map<std::string,std::string>::const_iterator i= uri.params.begin(); i!= uri.params.end(); ++i) {
        camera.SetProperty(i->first, i->second);
    }

    if( !camera.InitDriver(uri.scheme) ) {
        throw pangolin::VideoException("Couldn't start driver");
    }

    return camera;
}



inline CameraDevice OpenPangoCamera( const std::string& str_uri, const std::string& str_uri2)
{
    static CameraDriverRegisteryEntry<PangolinRpgVideoAdapterDriver> initialise("pangolin");

    CameraDevice camera;
    camera.SetProperty("URI0", str_uri);

    if(!str_uri2.empty()) {
        camera.SetProperty("URI1", str_uri2);
    }

    if( !camera.InitDriver("pangolin") ) {
        throw pangolin::VideoException("Couldn't start driver");
    }

    return camera;
}

inline bool PangolinRpgVideoAdapterDriver::Capture( std::vector<rpg::ImageWrapper>& vImages )
{
    // allocate images if necessary
    if( vImages.size() != m_numCams ){
        vImages.resize( m_numCams );
        for(unsigned i=0; i< m_numCams; ++i ) {
            vImages[i].Image = cv::Mat(pangovid[i].Height(), pangovid[i].Width(), CV_8UC1);
        }
    }

    for(unsigned i=0; i< m_numCams; ++i ) {
        pangovid[i].GrabNext(vImages[i].Image.data,true);
    }
    return true;
}

inline bool PangolinRpgVideoAdapterDriver::Init()
{
    m_numCams = 0;

    for(int i=0; i<MAX_CAMS; ++i) {
        std::stringstream ss;
        ss << "URI";
        ss << i;
        std::string uri = m_pPropertyMap->GetProperty<std::string>(ss.str());
        if(!uri.empty()) {
            ++m_numCams;
            pangovid[i].Open(uri);
        }
    }

    return m_numCams > 0;
}


#pragma once

#include <pangolin/pangolin.h>
#include <RPG/Devices/Camera/CameraDevice.h>
#include <RPG/Utils/GetPot>

CameraDevice OpenRpgCamera(int argc, char* argv[]);

//! Convenience method to load camera based on string URI containing PropertyMap and device.
CameraDevice OpenRpgCamera( const std::string& str_uri);

//! Convenience method to initialise camera based on string URI containing PropertyMap and device.
void InitRpgCamera( CameraDevice& camera, const std::string& str_uri);

//! Convenience method to retrieve pangolin video wrapped as RPG Video object.
CameraDevice OpenPangoCamera( const std::string& str_uri, const std::string& str_uri2 = "");

//! Convenience method to retrieve pangolin video wrapped as RPG Video object.
void InitPangoCamera( CameraDevice& camera, const std::string& str_uri, const std::string& str_uri2 = "");

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

const char USAGE[] =
"Usage:     program -idev <input> <options>\n"
"\n"
"where input device can be: FileReader Bumblebee2 etc\n"
"\n"
"Input Specific Options:\n"
"   FileReader:      -lfile <regular expression for left image channel>\n"
"                    -rfile <regular expression for right image channel>\n"
"                    -sdir  <directory where source images are located [default '.']>\n"
"                    -sf    <start frame [default 0]>\n"
"\n"
"General Options:    -lcmod <left camera model xml file>\n"
"                    -rcmod <right camera model xml file>\n"
"					 -gt <ground truth file> [not required]\n"
"\n"
"Example:\n"
"program  -idev FileReader  -lcmod lcmod.xml  -rcmod rcmod.xml  -lfile \"left.*pgm\"  -rfile \"right.*pgm\"\n\n";

inline CameraDevice OpenRpgCamera(int argc, char* argv[])
{
    if( argc < 2 ) {
        std::cout << USAGE;
        exit(0);
    }

    GetPot cl(argc,argv);

    CameraDevice camera;
    camera.SetProperty("NumChannels", 2);
    camera.SetProperty("DataSourceDir", cl.follow( ".", "-sdir"  ) );
    camera.SetProperty("Channel-0", cl.follow( ".*left.*", "-lfile" ) );
    camera.SetProperty("Channel-1", cl.follow( ".*right.*", "-rfile" ) );
    camera.SetProperty("StartFrame", cl.follow(0,"-sf") );
    camera.SetProperty("lcmod", cl.follow( "lcmod.xml", "-lcmod" ) );
    camera.SetProperty("rcmod", cl.follow( "rcmod.xml", "-rcmod" ) );
    camera.SetProperty("groundtruth", cl.follow( "", "-gt" ) );

    camera.InitDriver( cl.follow( "FileReader", "-idev" ) );
    return camera;
}

inline CameraDevice OpenRpgCamera( const std::string& str_uri)
{
    CameraDevice camera;
    InitRpgCamera(camera, str_uri);
    return camera;
}

inline void InitRpgCamera( CameraDevice& camera, const std::string& str_uri)
{
    pangolin::Uri uri = pangolin::ParseUri(str_uri);

    for(std::map<std::string,std::string>::const_iterator i= uri.params.begin(); i!= uri.params.end(); ++i) {
        camera.SetProperty(i->first, i->second);
    }

    if( !camera.InitDriver(uri.scheme) ) {
        throw pangolin::VideoException("Couldn't start driver");
    }
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

inline void InitPangoCamera( CameraDevice& camera, const std::string& str_uri, const std::string& str_uri2)
{
    static CameraDriverRegisteryEntry<PangolinRpgVideoAdapterDriver> initialise("pangolin");

    camera.SetProperty("URI0", str_uri);

    if(!str_uri2.empty()) {
        camera.SetProperty("URI1", str_uri2);
    }

    if( !camera.InitDriver("pangolin") ) {
        throw pangolin::VideoException("Couldn't start driver");
    }
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


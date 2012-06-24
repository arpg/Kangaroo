#include <pangolin/pangolin.h>
#include <RPG/Devices/Camera/CameraDevice.h>

//! Convenience method to load camera based on string URI containing PropertyMap and device.
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

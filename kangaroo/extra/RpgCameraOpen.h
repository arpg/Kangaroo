#pragma once

#include <HAL/Camera/CameraDevice.h>
#include <HAL/Utils/GetPot>

inline hal::Camera OpenRpgCamera( const std::string& str_uri)
{
    return hal::Camera(str_uri);
}

inline hal::Camera OpenRpgCamera(int argc, char* argv[])
{
    GetPot clArgs( argc, argv );    
    return hal::Camera(clArgs.follow("","-cam"));
}

inline void OpenRpgCamera(hal::Camera& camera, int argc, char* argv[])
{
    camera = OpenRpgCamera(argc,argv);
}

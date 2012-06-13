project( Kangaroo )
cmake_minimum_required(VERSION 2.8)

find_package( SimpleGui REQUIRED )
include_directories( ${SimpleGui_INCLUDE_DIRS} )

find_package( RPG REQUIRED )
include_directories( ${RPG_INCLUDE_DIRS} )

# git://github.com/stevenlovegrove/fiducials.git
find_package( Fiducials REQUIRED )
include_directories( ${Fiducials_INCLUDE_DIRS} )
link_directories(${Fiducials_LIBRARY_DIRS})

set( BuildImuTest true CACHE "" force )
if( BuildImuTest )
    add_executable( ImuTest ImuTest.cpp )
    target_link_libraries( ImuTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
endif()

set( BuildMeshTest true CACHE "" force )
if( BuildMeshTest )
    add_executable( MeshTest MeshTest.cpp )
    target_link_libraries( MeshTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
endif()

set( BuildTriggerTest true CACHE "" force )
if( BuildTriggerTest )
    add_executable( TriggerTest TriggerTest.cpp )
    target_link_libraries( TriggerTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
endif()

set( BuildStereoCalibration true CACHE "" force )
if( BuildStereoCalibration )
    add_executable( StereoCalibration StereoCalibration.cpp )
    target_link_libraries( StereoCalibration ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES} ${Fiducials_LIBRARIES} )
endif()

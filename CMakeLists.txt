project( Kangaroo )
cmake_minimum_required(VERSION 2.8)

find_package( SimpleGui REQUIRED )
include_directories( ${SimpleGui_INCLUDE_DIRS} )

find_package( RPG REQUIRED )
include_directories( ${RPG_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )

# git://github.com/stevenlovegrove/fiducials.git
find_package( Fiducials REQUIRED )
include_directories( ${Fiducials_INCLUDE_DIRS} )
link_directories(${Fiducials_LIBRARY_DIRS})

# git://github.com/stevenlovegrove/Sophus.git
find_package(Sophus REQUIRED)
include_directories( ${Sophus_INCLUDE_DIR} )
link_directories( ${Sophus_LIBRARY_DIRS} )

# git://github.com/stevenlovegrove/Pangolin.git
FIND_PACKAGE(Pangolin REQUIRED)
INCLUDE_DIRECTORIES(${Pangolin_INCLUDE_DIRS})
LINK_DIRECTORIES(${Pangolin_LIBRARY_DIRS})
LINK_LIBRARIES(${Pangolin_LIBRARIES})

FIND_PACKAGE(CUDA QUIET)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "-std=c++0x")
ENDIF()

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
    target_link_libraries( StereoCalibration ${RPG_LIBRARIES} ${SimpleGui_LIBRARIES} ${Fiducials_LIBRARIES} ${Sophus_LIBRARIES} ${OpenCV_LIBS} )
endif()

set( BuildDenseStereo true CACHE "" force )
if( BuildDenseStereo AND CUDA_FOUND)
    cuda_add_executable( DenseStereo DenseStereo.cpp )
    target_link_libraries( DenseStereo ${RPG_LIBRARIES} ${SimpleGui_LIBRARIES} ${Fiducials_LIBRARIES} ${Sophus_LIBRARIES} ${OpenCV_LIBS} )
endif()

set( BuildPangolinGLConsole true CACHE "" force )
if( BuildPangolinGLConsole AND CUDA_FOUND)
    cuda_add_executable( PangolinGLConsole PangolinGLConsole.cpp )
    target_link_libraries( PangolinGLConsole ${RPG_LIBRARIES} ${SimpleGui_LIBRARIES} ${Fiducials_LIBRARIES} ${Sophus_LIBRARIES} ${OpenCV_LIBS} )
endif()

set( BuildViconTest true CACHE "" force )
if( BuildViconTest AND CUDA_FOUND)
    cuda_add_executable( ViconTest ViconTest.cpp )
    target_link_libraries( ViconTest ${RPG_LIBRARIES} ${SimpleGui_LIBRARIES} ${Fiducials_LIBRARIES} ${Sophus_LIBRARIES} ${OpenCV_LIBS} vrpn )
endif()

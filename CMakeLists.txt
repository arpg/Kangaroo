project( Kangaroo )
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE( SimpleGui REQUIRED )
INCLUDE_DIRECTORIES( ${SimpleGui_INCLUDE_DIRS} )

# git://github.com/stevenlovegrove/fiducials.git
FIND_PACKAGE( Fiducials REQUIRED )
INCLUDE_DIRECTORIES( ${Fiducials_INCLUDE_DIRS} )
LINK_DIRECTORIES(${Fiducials_LIBRARY_DIRS})

# git://github.com/stevenlovegrove/Sophus.git
FIND_PACKAGE(Sophus REQUIRED)
INCLUDE_DIRECTORIES(${Sophus_INCLUDE_DIR})
LINK_DIRECTORIES(${Sophus_LIBRARY_DIRS})
LINK_LIBRARIES(${Sophus_LIBRARIES})

# git://github.com/stevenlovegrove/Pangolin.git
FIND_PACKAGE(Pangolin REQUIRED)
INCLUDE_DIRECTORIES(${Pangolin_INCLUDE_DIRS})
LINK_DIRECTORIES(${Pangolin_LIBRARY_DIRS})
LINK_LIBRARIES(${Pangolin_LIBRARIES})

FIND_PACKAGE( RPG REQUIRED )
INCLUDE_DIRECTORIES( ${RPG_INCLUDE_DIRS} )
LINK_LIBRARIES(${RPG_LIBRARIES})

FIND_PACKAGE( OpenCV REQUIRED )
LINK_LIBRARIES(${OpenCV_LIBS})


IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "-std=c++0x")
ENDIF()

set( BuildImuTest true CACHE "" force )
if( BuildImuTest )
    add_executable( ImuTest ImuTest.cpp )
    target_link_libraries( ImuTest ${SimpleGui_LIBRARIES} )
endif()

set( BuildMeshTest true CACHE "" force )
if( BuildMeshTest )
    add_executable( MeshTest MeshTest.cpp )
    target_link_libraries( MeshTest ${SimpleGui_LIBRARIES} )
endif()

set( BuildTriggerTest true CACHE "" force )
if( BuildTriggerTest )
    add_executable( TriggerTest TriggerTest.cpp )
    target_link_libraries( TriggerTest ${SimpleGui_LIBRARIES} )
endif()

set( BuildStereoCalibration true CACHE "" force )
if( BuildStereoCalibration )
    add_executable( StereoCalibration StereoCalibration.cpp )
    target_link_libraries( StereoCalibration ${SimpleGui_LIBRARIES} ${Fiducials_LIBRARIES} )
endif()

set( BuildPangolinGLConsole true CACHE "" force )
if( BuildPangolinGLConsole )
    add_executable( PangolinGLConsole PangolinGLConsole.cpp )
endif()

set( BuildViconTest true CACHE "" force )
if( BuildViconTest )
    add_executable( ViconTest ViconTest.cpp )
    target_link_libraries( ViconTest vrpn )
endif()

set( BuildCameraViewer true CACHE "" force )
if( BuildCameraViewer )
    add_executable( CameraViewer CameraViewer.cpp )
    target_link_libraries( CameraViewer )
endif()



FIND_PACKAGE(CUDA QUIET)

if(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS "-arch=sm_20 --compiler-options \"-std=c++03\"" )

  set( BuildDenseStereo true CACHE "" force )
  if( BuildDenseStereo )
      cuda_add_executable( DenseStereo DenseStereo.cpp kernel.h kernel.cu )
  endif()

  set( BuildStereoViewer true CACHE "" force )
  if( BuildStereoViewer )
      cuda_add_executable( StereoViewer StereoViewer.cpp kernel.h kernel.cu )
  endif()
endif(CUDA_FOUND)

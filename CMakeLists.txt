PROJECT( Kangaroo )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/")

# git://github.com/stevenlovegrove/fiducials.git
find_package( Fiducials REQUIRED )
include_directories( ${Fiducials_INCLUDE_DIRS} )
LINK_DIRECTORIES(${Fiducials_LIBRARY_DIRS})
LINK_LIBRARIES(${Fiducials_LIBRARIES})

# git://github.com/stevenlovegrove/Sophus.git
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})
LINK_DIRECTORIES(${Sophus_LIBRARY_DIRS})
LINK_LIBRARIES(${Sophus_LIBRARIES})

# git://github.com/stevenlovegrove/Pangolin.git
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
LINK_DIRECTORIES(${Pangolin_LIBRARY_DIRS})
LINK_LIBRARIES(${Pangolin_LIBRARIES})

find_package( RPG REQUIRED )
include_directories( ${RPG_INCLUDE_DIRS} )
LINK_LIBRARIES(${RPG_LIBRARIES})

#FIND_PACKAGE( NODE REQUIRED )
#find_package( ZeroMQ REQUIRED )
find_package( Protobuf REQUIRED )
set(NODE_INCLUDE_DIRS "/Users/slovegrove/code/RpgDev/Hermes1.0")
include_directories( ${ZeroMQ_INCLUDE_DIRS} )
include_directories( ${PROTOBUF_INCLUDE_DIRS} )
include_directories( ${NODE_INCLUDE_DIRS} )
PROTOBUF_GENERATE_CPP( PROTO_SRCS PROTO_HDRS Messages.proto )


SET(COMMON_SRC "")

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "-std=c++0x -g")
ENDIF()

FIND_PACKAGE( SimpleGui QUIET )

if(SimpleGui_FOUND)
set( BuildImuTest true CACHE "" force )
    include_directories( ${SimpleGui_INCLUDE_DIRS} )

    if( BuildImuTest )
        add_executable( ImuTest ImuTest.cpp ${COMMON_SRC})
        target_link_libraries( ImuTest ${SimpleGui_LIBRARIES} )
    endif()

    set( BuildMeshTest true CACHE "" force )
    if( BuildMeshTest )
        add_executable( MeshTest MeshTest.cpp ${COMMON_SRC})
        target_link_libraries( MeshTest ${SimpleGui_LIBRARIES} )
    endif()
endif()

set( BuildTriggerTest false CACHE "" force )
if( BuildTriggerTest )
    add_executable( TriggerTest TriggerTest.cpp ${COMMON_SRC})
    target_link_libraries( TriggerTest ${SimpleGui_LIBRARIES} )
endif()

set( BuildStereoCalibration true CACHE "" force )
if( BuildStereoCalibration )
    add_executable( StereoCalibration StereoCalibration.cpp ${COMMON_SRC})
    target_link_libraries( StereoCalibration ${Fiducials_LIBRARIES} )
endif()

set( BuildViconTest true CACHE "" force )
if( BuildViconTest )
    add_executable( ViconTest ViconTest.cpp ${COMMON_SRC})
    target_link_libraries( ViconTest vrpn )
endif()

set( BuildCameraViewer true CACHE "" force )
if( BuildCameraViewer )
    add_executable( CameraViewer CameraViewer.cpp ${COMMON_SRC})
    target_link_libraries( CameraViewer )
endif()



FIND_PACKAGE(CUDA QUIET)

if(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS "-arch=sm_20 --compiler-options \"-std=c++03\"" )

  if(NOT CUDA_npp_LIBRARY)
    # Hack for old FindCUDA to locate libnpp.so
    set(CUDA_npp_LIBRARY "/usr/local/cuda/lib64/libnpp.so")
  endif()

  set( BuildCheckMemory true CACHE "" force )
  if( BuildCheckMemory )
      cuda_add_executable( CheckMemory CheckMemory.cpp)
  endif()

  set( BuildStereoViewer true CACHE "" force )
  if( BuildStereoViewer )
      cuda_add_executable( StereoViewer StereoViewer.cpp kernel.h kernel.cu ${COMMON_SRC})
  endif()

  set( BuildDenseStereo true CACHE "" force )
  if( BuildDenseStereo )
      cuda_add_executable( DenseStereo DenseStereo.cpp kernel.h kernel.cu ScanlineRectify.h CudaImage.h ${COMMON_SRC})
      target_link_libraries( DenseStereo ${CUDA_npp_LIBRARY} )
  endif()

  set( BuildBilateralFilter true CACHE "" force )
  if( BuildBilateralFilter )
      cuda_add_executable( BilateralFilter BilateralFilter.cpp kernel.h kernel.cu CudaImage.h ${COMMON_SRC})
      target_link_libraries( BilateralFilter ${CUDA_npp_LIBRARY} )
  endif()
endif(CUDA_FOUND)

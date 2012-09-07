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

# Find Scenegraph
FIND_PACKAGE(SceneGraph REQUIRED)
INCLUDE_DIRECTORIES(${SceneGraph_INCLUDE_DIRS})
LINK_LIBRARIES(${SceneGraph_LIBRARIES})

find_package( RPG REQUIRED )
include_directories( ${RPG_INCLUDE_DIRS} )
LINK_LIBRARIES(${RPG_LIBRARIES})

#FIND_PACKAGE( NODE REQUIRED )
#find_package( ZeroMQ REQUIRED )
find_package( Protobuf REQUIRED )
set(NODE_INCLUDE_DIRS "~/Code/Dev/Hermes1.0")
include_directories( ${ZeroMQ_INCLUDE_DIRS} )
include_directories( ${PROTOBUF_INCLUDE_DIRS} )
include_directories( ${NODE_INCLUDE_DIRS} )
PROTOBUF_GENERATE_CPP( PROTO_SRCS PROTO_HDRS Messages.proto )


LIST(APPEND COMMON_HEADERS
    common/DisplayUtils.h common/RpgCameraOpen.h common/ImageSelect.h
    common/LeastSquaresBlockHelpers.h common/ViconTracker.h
    common/AssimpVboExport.h
)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "-std=c++11 -g")
ENDIF()

set( BuildOddsAndEnds true CACHE "" force )
if( BuildOddsAndEnds )
    add_executable( OddsAndEnds ${COMMON_SRC} OddsAndEnds.cpp)
    target_link_libraries( OddsAndEnds )
endif()

set( BuildCameraViewer true CACHE "" force )
if( BuildCameraViewer )
    add_executable( CameraViewer ${COMMON_SRC} CameraViewer.cpp)
    target_link_libraries( CameraViewer )
endif()

set( BuildTriggerTest false CACHE "" force )
if( BuildTriggerTest )
    add_executable( TriggerTest ${COMMON_SRC} TriggerTest.cpp)
    target_link_libraries( TriggerTest ${SimpleGui_LIBRARIES} )
endif()

set( BuildStereoCalibration true CACHE "" force )
if( BuildStereoCalibration )
    add_executable( StereoCalibration ${COMMON_SRC} StereoCalibration.cpp)
    target_link_libraries( StereoCalibration ${Fiducials_LIBRARIES} )
endif()

set( BuildViconTest true CACHE "" force )
if( BuildViconTest )
    add_executable( ViconTest ${COMMON_SRC} ViconTest.cpp)
    target_link_libraries( ViconTest vrpn )
endif()

set( BuildViconAlign true CACHE "" force )
if( BuildViconAlign )
    add_executable( ViconAlign ${COMMON_SRC} ViconAlign.cpp)
    target_link_libraries( ViconAlign vrpn )
endif()

set( BuildGpsConvert true CACHE "" force )
if( BuildGpsConvert )
    add_executable( GpsConvert ${COMMON_SRC} GpsConvert.cpp)
    target_link_libraries( GpsConvert Geographic )
endif()


find_package( Kangaroo )
if(Kangaroo_FOUND)
    include_directories( ${Kangaroo_INCLUDE_DIRS} )
    link_libraries(${Kangaroo_LIBRARIES} vrpn)

    set( BuildCheckMemory true CACHE "" force )
    if( BuildCheckMemory )
      add_executable( CheckMemory CheckMemory.cpp)
    endif()

    set( BuildCudaSumSpeed true CACHE "" force )
    if( BuildCudaSumSpeed )
      add_executable( CudaSumSpeed CudaSumSpeed.cpp)
    endif()

    set( BuildStereoViewer true CACHE "" force )
    if( BuildStereoViewer )
      add_executable( StereoViewer StereoViewer.cpp ${COMMON_HEADERS})
    endif()

    set( BuildSimpleStereo true CACHE "" force )
    if( BuildSimpleStereo )
      add_executable( SimpleStereo SimpleStereo.cpp ${COMMON_HEADERS})
    endif()

    set( BuildDenseStereo true CACHE "" force )
    if( BuildDenseStereo )
      add_executable( DenseStereo DenseStereo.cpp ${COMMON_HEADERS})
    endif()

    set( BuildKinectHeightmap true CACHE "" force )
    if( BuildKinectHeightmap )
      add_executable( KinectHeightmap KinectHeightmap.cpp ${COMMON_HEADERS})
    endif()

    set( BuildBilateralFilter true CACHE "" force )
    if( BuildBilateralFilter )
      add_executable( BilateralFilter BilateralFilter.cpp ${COMMON_HEADERS})
    endif()

    set( BuildManhatten true CACHE "" force )
    if( BuildManhatten )
      add_executable( Manhatten Manhatten.cpp ${COMMON_HEADERS})
    endif()
endif()

project( Kangaroo )
cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/")

# git://github.com/stevenlovegrove/fiducials.git
find_package( Fiducials REQUIRED )
include_directories( ${Fiducials_INCLUDE_DIRS} )
link_directories(${Fiducials_LIBRARY_DIRS})
link_libraries(${Fiducials_LIBRARIES})

# git://github.com/stevenlovegrove/Sophus.git
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})
link_directories(${Sophus_LIBRARY_DIRS})
link_libraries(${Sophus_LIBRARIES})

# git://github.com/stevenlovegrove/Pangolin.git
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
link_directories(${Pangolin_LIBRARY_DIRS})
link_libraries(${Pangolin_LIBRARIES})

# Find Scenegraph
find_package(SceneGraph REQUIRED)
include_directories(${SceneGraph_INCLUDE_DIRS})
link_libraries(${SceneGraph_LIBRARIES})

# https://github.com/obiou/FPL.git
find_package(CCAMERAMODEL QUIET)

find_package( RPG REQUIRED )
include_directories( ${RPG_INCLUDE_DIRS} )
link_libraries(${RPG_LIBRARIES})

#git clone https://ceres-solver.googlesource.com/ceres-solver
find_package( CeresSolver QUIET)

#FIND_PACKAGE( NODE REQUIRED )
#find_package( ZeroMQ REQUIRED )
find_package( Protobuf REQUIRED )
set(NODE_INCLUDE_DIRS "~/Code/Dev/Hermes1.0")
include_directories( ${ZeroMQ_INCLUDE_DIRS} )
include_directories( ${PROTOBUF_INCLUDE_DIRS} )
include_directories( ${NODE_INCLUDE_DIRS} )
protobuf_generate_cpp( PROTO_SRCS PROTO_HDRS Messages.proto )


list(APPEND COMMON_HEADERS
    common/DisplayUtils.h common/RpgCameraOpen.h common/ImageSelect.h
    common/LeastSquaresBlockHelpers.h common/ViconTracker.h
    common/AssimpVboExport.h
)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-std=c++11 -g")
endif()

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

set( BuildTriggerTest true CACHE "" force )
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

set( BuildStereoCalibrationOpenCV true CACHE "" force )
if( BuildStereoCalibrationOpenCV )
    add_executable( StereoCalibrationOpenCV ${COMMON_SRC} StereoCalibrationOpenCV.cpp)
endif()

set( BuildStereoRectifyOpenCV true CACHE "" force )
if( BuildStereoRectifyOpenCV )
    add_executable( StereoRectifyOpenCV ${COMMON_SRC} StereoRectifyOpenCV.cpp)
endif()

#if(CeresSolver_FOUND)
#    include_directories( ${CeresSolver_INCLUDE_DIRS} )
#    set( BuildPoseGraph true CACHE "" force )
#    if( BuildPoseGraph )
#      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
#      add_executable( PoseGraph PoseGraph.cpp)
#      target_link_libraries( PoseGraph ${CeresSolver_LIBRARIES} )
#    endif()
#endif()

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

    set( BuildKinectTrack true CACHE "" force )
    if( BuildKinectTrack )
      add_executable( KinectTrack KinectTrack.cpp ${COMMON_HEADERS})
    endif()

    set( BuildKinectHeightmap true CACHE "" force )
    if( BuildKinectHeightmap )
      add_executable( KinectHeightmap KinectHeightmap.cpp ${COMMON_HEADERS})
    endif()

    set( BuildBilateralFilter true CACHE "" force )
    if( BuildBilateralFilter )
      add_executable( BilateralFilter BilateralFilter.cpp ${COMMON_HEADERS})
    endif()

    set( BuildGuidedFilter true CACHE "" force )
    if( BuildGuidedFilter )
      add_executable( GuidedFilter GuidedFilter.cpp ${COMMON_HEADERS})
    endif()

    set( BuildManhatten true CACHE "" force )
    if( BuildManhatten )
      add_executable( Manhatten Manhatten.cpp ${COMMON_HEADERS})
    endif()

    set( BuildDenoising true CACHE "" force )
    if( BuildDenoising )
      add_executable( Denoising Denoising.cpp ${COMMON_HEADERS})
    endif()

    set( BuildDeconvolution true CACHE "" force )
    if( BuildDeconvolution )
      add_executable( Deconvolution Deconvolution.cpp ${COMMON_HEADERS})
    endif()

    set( BuildInpainting true CACHE "" force )
    if( BuildInpainting )
      add_executable( Inpainting Inpainting.cpp ${COMMON_HEADERS})
    endif()

    set( BuildFeatures true CACHE "" force )
    if( BuildFeatures )
      add_executable( Features Features.cpp ${COMMON_HEADERS})
    endif()

endif()

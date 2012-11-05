cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/")

# git clone git://github.com/stevenlovegrove/fiducials.git
find_package( Fiducials REQUIRED )
include_directories( ${Fiducials_INCLUDE_DIRS} )
link_directories(${Fiducials_LIBRARY_DIRS})
link_libraries(${Fiducials_LIBRARIES})

# git clone git://github.com/stevenlovegrove/Sophus.git
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})
link_directories(${Sophus_LIBRARY_DIRS})
link_libraries(${Sophus_LIBRARIES})

# git clone git://github.com/stevenlovegrove/Pangolin.git
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
link_directories(${Pangolin_LIBRARY_DIRS})
link_libraries(${Pangolin_LIBRARIES})

# git clone http://robotics.gwu.edu/git/SceneGraph
find_package(SceneGraph REQUIRED)
include_directories(${SceneGraph_INCLUDE_DIRS})
link_libraries(${SceneGraph_LIBRARIES})

# git clone https://github.com/obiou/FPL.git
find_package(CCAMERAMODEL QUIET)

# git clone http://robotics.gwu.edu/git/RPG
find_package( RPG REQUIRED )
include_directories( ${RPG_INCLUDE_DIRS} )
link_libraries(${RPG_LIBRARIES})

# git clone https://ceres-solver.googlesource.com/ceres-solver
find_package( CeresSolver QUIET)

# Other libraries without find script
link_libraries( vrpn Geographic)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-std=c++11 -g")
endif()

add_executable( CameraViewer CameraViewer.cpp)
add_executable( StereoCalibration StereoCalibration.cpp)
add_executable( ViconTest ViconTest.cpp)
add_executable( ViconAlign ViconAlign.cpp)
add_executable( GpsConvert GpsConvert.cpp)
add_executable( StereoCalibrationOpenCV StereoCalibrationOpenCV.cpp)
add_executable( StereoRectifyOpenCV StereoRectifyOpenCV.cpp)

find_package( NODE QUIET )
if(NODE_FOUND)
    include_directories( ${NODE_INCLUDE_DIRS} )
    link_libraries( ${NODE_LIBRARIES} )
    include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

    if(CMAKE_COMPILER_IS_GNUCXX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOS_OBJECT_USE_OBJC=0")
    endif()

    message(STATUS "bin dir ${CMAKE_CURRENT_BINARY_DIR}" )

    set( BuildNodeTest true CACHE "" force )
    if( BuildNodeTest )
        PROTOBUF_GENERATE_CPP( PROTO_SRCS PROTO_HDRS NodeTestMessages.proto )
        add_executable( NodeTest ${PROTO_SRCS} ${PROTO_HDRS} NodeTest.cpp)
    endif()
endif()


if(CeresSolver_FOUND)
    include_directories( ${CeresSolver_INCLUDE_DIRS} )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

    add_executable( PoseGraphGps PoseGraphGps.cpp)
    target_link_libraries( PoseGraphGps Geographic )
    target_link_libraries( PoseGraphGps ${CeresSolver_LIBRARIES} )

    add_executable( PoseGraphVicon PoseGraphVicon.cpp)
    target_link_libraries( PoseGraphVicon ${CeresSolver_LIBRARIES} )
endif()

find_package( Kangaroo )
if(Kangaroo_FOUND)
    include_directories( ${Kangaroo_INCLUDE_DIRS} )
    link_libraries(${Kangaroo_LIBRARIES})

    add_executable( CheckMemory CheckMemory.cpp)
    add_executable( CudaSumSpeed CudaSumSpeed.cpp)
    add_executable( StereoViewer StereoViewer.cpp)
    add_executable( SimpleStereo SimpleStereo.cpp)
    add_executable( DenseStereo DenseStereo.cpp)
    add_executable( KinectHeightmap KinectHeightmap.cpp)
    add_executable( BilateralFilter BilateralFilter.cpp)
    add_executable( GuidedFilter GuidedFilter.cpp)
    add_executable( Manhatten Manhatten.cpp)
    add_executable( Denoising Denoising.cpp)
    add_executable( Deconvolution Deconvolution.cpp)
    add_executable( Inpainting Inpainting.cpp)
    add_executable( Features Features.cpp)
    add_executable( Raycast Raycast.cpp)
    add_executable( SdfFusion SdfFusion.cpp)
    add_executable( MocapMultiDepthFusion MocapMultiDepthFusion.cpp)

  if(CeresSolver_FOUND)
    include_directories( ${CeresSolver_INCLUDE_DIRS} )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

    add_executable( OddsAndEnds OddsAndEnds.cpp MarchingCubes.cpp)
    target_link_libraries( OddsAndEnds ${CeresSolver_LIBRARIES} )

    add_executable( KinectTrack KinectTrack.cpp)
    target_link_libraries( KinectTrack ${CeresSolver_LIBRARIES} )

    add_executable( KinectFusion KinectFusion.cpp MarchingCubes.cpp)
    target_link_libraries( KinectFusion ${CeresSolver_LIBRARIES} )
  endif()

endif()

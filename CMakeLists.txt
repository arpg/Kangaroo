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


LIST(APPEND COMMON_SRC
    common/DisplayUtils.h common/RpgCameraOpen.h common/ImageSelect.h
    common/LeastSquaresBlockHelpers.h common/ViconTracker.h
    common/AssimpVboExport.h
)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "-std=c++0x -g")
ENDIF()

FIND_PACKAGE( SimpleGui QUIET )

if(SimpleGui_FOUND)
set( BuildImuTest true CACHE "" force )
    include_directories( ${SimpleGui_INCLUDE_DIRS} )

    if( BuildImuTest )
        add_executable( ImuTest ${COMMON_SRC} ImuTest.cpp)
        target_link_libraries( ImuTest ${SimpleGui_LIBRARIES} )
    endif()

    set( BuildMeshTest true CACHE "" force )
    if( BuildMeshTest )
        add_executable( MeshTest ${COMMON_SRC} MeshTest.cpp)
        target_link_libraries( MeshTest ${SimpleGui_LIBRARIES} )
    endif()
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

set( BuildCameraViewer true CACHE "" force )
if( BuildCameraViewer )
    add_executable( CameraViewer ${COMMON_SRC} CameraViewer.cpp)
    target_link_libraries( CameraViewer )
endif()



FIND_PACKAGE(CUDA QUIET)

if(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS "-arch=sm_20 --compiler-options \"-std=c++03\"" )
  LIST(APPEND CU_COMMON_SRC cu/all.h cu/Image.h cu/Mat.h cu/MatUtils.h cu/sampling.h cu/pixel_convert.h cu/patch_score.h cu/reweighting.h)
  LIST(APPEND CU_COMMON_SRC cu/cu_anaglyph.cu cu/cu_bilateral.cu cu/cu_convert.cu cu/cu_index_buffer.cu
                            cu/cu_model_refinement.cu cu/cu_dense_stereo.cu cu/cu_lookup_warp.cu
                            cu/cu_median.cu cu/cu_normals.cu cu/cu_openni.cu cu/cu_plane_fit.cu
                            cu/cu_resample.cu cu/cu_heightmap.cu)

  if(NOT CUDA_npp_LIBRARY)
    # Hack for old FindCUDA to locate libnpp.so
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      set(CUDA_npp_LIBRARY "/usr/local/cuda/lib/libnpp.dylib")
    else()
      set(CUDA_npp_LIBRARY "/usr/local/cuda/lib64/libnpp.so")
    endif()
  endif()

  set( BuildCheckMemory true CACHE "" force )
  if( BuildCheckMemory )
      cuda_add_executable( CheckMemory CheckMemory.cpp)
  endif()

  set( BuildStereoViewer true CACHE "" force )
  if( BuildStereoViewer )
      cuda_add_executable( StereoViewer ${CU_COMMON_SRC} ${COMMON_SRC} StereoViewer.cpp)
  endif()

  set( BuildDenseStereo true CACHE "" force )
  if( BuildDenseStereo )
      cuda_add_executable( DenseStereo ${CU_COMMON_SRC} ${COMMON_SRC} DenseStereo.cpp common/ScanlineRectify.h common/HeightmapFusion.h)
      target_link_libraries( DenseStereo ${CUDA_npp_LIBRARY} )
  endif()

  set( BuildKinectHeightmap true CACHE "" force )
  if( BuildKinectHeightmap )
      cuda_add_executable( KinectHeightmap ${CU_COMMON_SRC} ${COMMON_SRC} KinectHeightmap.cpp common/ScanlineRectify.h common/HeightmapFusion.h )
      target_link_libraries( KinectHeightmap ${CUDA_npp_LIBRARY} vrpn )
  endif()

  set( BuildBilateralFilter true CACHE "" force )
  if( BuildBilateralFilter )
      cuda_add_executable( BilateralFilter ${CU_COMMON_SRC} ${COMMON_SRC} BilateralFilter.cpp )
      target_link_libraries( BilateralFilter ${CUDA_npp_LIBRARY} )
  endif()
endif(CUDA_FOUND)

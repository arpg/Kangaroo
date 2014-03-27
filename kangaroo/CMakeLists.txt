cmake_minimum_required( VERSION 2.8 )
project( Kangaroo )

set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR})
string( TOLOWER ${PROJECT_NAME} LIBRARY_NAME )

################################################################################

list(APPEND SRC_H
    BoundedVolume.h   MarchingCubesTables.h cu_deconvolution.h    cu_painting.h             kangaroo.h
    BoundingBox.h     Mat.h                 cu_dense_stereo.h     cu_plane_fit.h            launch_utils.h
    CostVolElem.h     MatUtils.h            cu_depth_tools.h      cu_raycast.h              patch_score.h
    CudaTimer.h       Pyramid.h             cu_heightmap.h        cu_remap.h                pixel_convert.h
    Divergence.h      Sdf.h                 cu_index_buffer.h     cu_resample.h             reduce.h
    Image.h           Volume.h              cu_integral_image.h   cu_rof_denoising.h        reweighting.h
    ImageApron.h      cu_anaglyph.h         cu_lookup_warp.h      cu_sdffusion.h            sampling.h
    ImageIntrinsics.h cu_bilateral.h        cu_manhattan.h        cu_segment_test.h         variational.h
    ImageKeyframe.h   cu_blur.h             cu_median.h           cu_semi_global_matching.h
    InvalidValue.h    cu_census.h           cu_model_refinement.h cu_tgv.h
    LeastSquareSum.h  cu_convert.h          cu_normals.h          disparity.h
    MarchingCubes.h   cu_convolution.h      cu_operations.h       hamming_distance.h
)

list(APPEND SRC_CU
    cu_anaglyph.cu cu_bilateral.cu cu_convert.cu cu_index_buffer.cu
    cu_model_refinement.cu cu_dense_stereo.cu cu_lookup_warp.cu
    cu_median.cu cu_normals.cu cu_depth_tools.cu cu_plane_fit.cu
    cu_resample.cu cu_heightmap.cu cu_blur.cu cu_operations.cu
    cu_census.cu cu_semi_global_matching.cu
    cu_manhattan.cu cu_integral_image.cu
    cu_convolution.cu cu_deconvolution.cu
    cu_rof_denoising.cu cu_tgv.cu
    cu_segment_test.cu
    cu_painting.cu cu_remap.cu
    cu_raycast.cu cu_sdffusion.cu
    MarchingCubes.cu
)

################################################################################
# Find required dependencies

# Find CUDA toolkit. Thrust and NPP are bundled with CUDA toolkit.
find_package(CUDA REQUIRED)
list( APPEND USER_INC ${CUDA_INCLUDE_DIRS} )
set(HAVE_THRUST 1)
set(HAVE_NPP 1)

# Dont propogate host flags into CUDA environment.
set(CUDA_PROPAGATE_HOST_FLAGS off)

# Generate code for 2.0 and 3.0 CUDA compute architectures (TODO: Verify this works)
set(CUDA_NVCC_FLAGS "-use_fast_math -gencode arch=compute_30,code=sm_30 -gencode arch=compute_20,code=sm_20" )

# Make sure that NPP is found
if(NOT CUDA_npp_LIBRARY)
    # TODO: Fix FindCUDA or make this more robust
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/libnppc.dylib;${CUDA_TOOLKIT_ROOT_DIR}/lib/libnppi.dylib;${CUDA_TOOLKIT_ROOT_DIR}/lib/libnpps.dylib")
    elseif(WIN32)
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/nppc.lib;${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/nppi.lib;${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/npps.lib")
    elseif(WIN64)
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nppc.lib;${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nppi.lib;${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/npps.lib")
    else()
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnpps.so;${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppi.so;${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnpps.so")
    endif()
endif()
list(APPEND LINK_LIBS ${CUDA_npp_LIBRARY} )

find_package( Eigen3 QUIET )
if(EIGEN3_FOUND)
    set(HAVE_EIGEN 1)
    list( APPEND USER_INC ${EIGEN3_INCLUDE_DIR} )
endif()

find_package( ASSIMP QUIET )
if(ASSIMP_FOUND)
    set(HAVE_ASSIMP 1)
    list(APPEND INTERNAL_INC ${ASSIMP_INCLUDE_DIR})
    list(APPEND LINK_LIBS ${ASSIMP_LIBRARIES} )
endif()

find_package( OpenCV2 QUIET )
if(OpenCV2_FOUND)
    set(HAVE_OPENCV 1)
    list(APPEND USER_INC ${OpenCV2_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${OpenCV2_LIBRARIES} )
endif()

# So that we can find config.h
list( APPEND LIB_INC_DIR "${CMAKE_CURRENT_BINARY_DIR}/..;${CMAKE_CURRENT_SOURCE_DIR}/.." )

#######################################################
## Create configure file for inclusion in library

CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h"
)

################################################################################
# Include collected includes / libraries
include_directories( ${USER_INC} )
include_directories( ${LIB_INC_DIR} )
include_directories( ${INTERNAL_INC} )
cuda_add_library( ${LIBRARY_NAME} ${SRC_H} ${SRC_CU} )
target_link_libraries(${LIBRARY_NAME} ${LINK_LIBS})

## Generate symbol export helper header on MSVC
if(MSVC)
    string(TOUPPER ${LIBRARY_NAME} LIBRARY_NAME_CAPS)
    include(GenerateExportHeader)
    generate_export_header( ${LIBRARY_NAME}
        BASE_NAME ${LIBRARY_NAME_CAPS}
        EXPORT_MACRO_NAME ${LIBRARY_NAME_CAPS}_EXPORT
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}_export.h"
        STATIC_DEFINE ${LIBRARY_NAME_CAPS}_BUILT_AS_STATIC
    )
endif()

#######################################################

# This relative path allows installed files to be relocatable.
set( CMAKECONFIG_INSTALL_DIR "lib/cmake/${PROJECT_NAME}" )
file( RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}"
    "${CMAKE_INSTALL_PREFIX}/include" )

# Export library for easy inclusion from other cmake projects.
export( TARGETS ${LIBRARY_NAME}
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake" )

# Version information
configure_file("${PROJECT_NAME}ConfigVersion.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

# Build tree config
set( EXPORT_LIB_INC_DIR "${LIB_INC_DIR}" )
CONFIGURE_FILE( "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY IMMEDIATE )

# Install tree config
set( EXPORT_LIB_INC_DIR "\${${PROJECT_NAME}_CMAKE_DIR}/${REL_INCLUDE_DIR}" )
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake" @ONLY )

# Add package to CMake package registery for use from the build tree
export( PACKAGE ${PROJECT_NAME} )

#######################################################
## Install headers / targets

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.h"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(FILES ${SRC_H}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(TARGETS ${LIBRARY_NAME}
  EXPORT "${PROJECT_NAME}Targets"
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

#######################################################
## Install CMake config

INSTALL(
    FILES "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKECONFIG_INSTALL_DIR} )

install( EXPORT "${PROJECT_NAME}Targets" DESTINATION ${CMAKECONFIG_INSTALL_DIR} )

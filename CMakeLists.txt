cmake_minimum_required( VERSION 2.8 )
project( Kangaroo )
set( LIBRARY_NAME kangaroo )

set(KANGAROO_VERSION_MAJOR 0)
set(KANGAROO_VERSION_MINOR 1)

set(VERSION ${KANGAROO_VERSION_MAJOR}.${KANGAROO_VERSION_MINOR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
include( install_package )

option(BUILD_APPLICATIONS "Build Applications" ON)
option(BUILD_SHARED_LIBS "Build Shared Library" ON)

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

# Platform configuration vars
include(SetPlatformVars)

# Disable unknown pragma warnings - pragma unrolls are for NVCC.
if(MSVC)
    add_definitions( "/wd4068" )
else()
    add_definitions( "-Wno-unknown-pragmas" )
endif()

################################################################################
set( INCDIR "${CMAKE_CURRENT_SOURCE_DIR}/include/kangaroo" )
set( SRC "${CMAKE_CURRENT_SOURCE_DIR}/src" )


list(APPEND SRC_H
    ${INCDIR}/BoundedVolume.h
    ${INCDIR}/MarchingCubesTables.h
    ${INCDIR}/cu_deconvolution.h
    ${INCDIR}/cu_painting.h
    ${INCDIR}/kangaroo.h
    ${INCDIR}/BoundingBox.h
    ${INCDIR}/Mat.h
    ${INCDIR}/cu_dense_stereo.h
    ${INCDIR}/cu_plane_fit.h
    ${INCDIR}/launch_utils.h
    ${INCDIR}/CostVolElem.h
    ${INCDIR}/MatUtils.h
    ${INCDIR}/cu_depth_tools.h
    ${INCDIR}/cu_raycast.h
    ${INCDIR}/patch_score.h
    ${INCDIR}/CudaTimer.h
    ${INCDIR}/Pyramid.h
    ${INCDIR}/cu_heightmap.h
    ${INCDIR}/cu_remap.h
    ${INCDIR}/pixel_convert.h
    ${INCDIR}/Divergence.h
    ${INCDIR}/Sdf.h
    ${INCDIR}/cu_index_buffer.h
    ${INCDIR}/cu_resample.h
    ${INCDIR}/reduce.h
    ${INCDIR}/Image.h
    ${INCDIR}/Volume.h
    ${INCDIR}/cu_integral_image.h
    ${INCDIR}/cu_rof_denoising.h
    ${INCDIR}/reweighting.h
    ${INCDIR}/ImageApron.h
    ${INCDIR}/cu_anaglyph.h
    ${INCDIR}/cu_lookup_warp.h
    ${INCDIR}/cu_sdffusion.h
    ${INCDIR}/sampling.h
    ${INCDIR}/ImageIntrinsics.h
    ${INCDIR}/cu_bilateral.h
    ${INCDIR}/cu_manhattan.h
    ${INCDIR}/cu_segment_test.h
    ${INCDIR}/variational.h
    ${INCDIR}/ImageKeyframe.h
    ${INCDIR}/cu_blur.h
    ${INCDIR}/cu_median.h
    ${INCDIR}/cu_semi_global_matching.h
    ${INCDIR}/Memory.h
    ${INCDIR}/InvalidValue.h
    ${INCDIR}/cu_census.h
    ${INCDIR}/cu_model_refinement.h
    ${INCDIR}/cu_tgv.h
    ${INCDIR}/platform.h
    ${INCDIR}/LeastSquareSum.h
    ${INCDIR}/cu_convert.h
    ${INCDIR}/cu_normals.h
    ${INCDIR}/disparity.h
    ${INCDIR}/MarchingCubes.h
    ${INCDIR}/cu_convolution.h
    ${INCDIR}/cu_operations.h
    ${INCDIR}/hamming_distance.h
)

list(APPEND SRC_CU
    ${SRC}/cu_anaglyph.cu
    ${SRC}/cu_bilateral.cu
    ${SRC}/cu_convert.cu
    ${SRC}/cu_index_buffer.cu
    ${SRC}/cu_model_refinement.cu
    ${SRC}/cu_dense_stereo.cu
    ${SRC}/cu_lookup_warp.cu
    ${SRC}/cu_median.cu
    ${SRC}/cu_normals.cu
    ${SRC}/cu_depth_tools.cu
    ${SRC}/cu_plane_fit.cu
    ${SRC}/cu_resample.cu
    ${SRC}/cu_heightmap.cu
    ${SRC}/cu_blur.cu
    ${SRC}/cu_operations.cu
    ${SRC}/cu_census.cu
    ${SRC}/cu_semi_global_matching.cu
    ${SRC}/cu_manhattan.cu
    ${SRC}/cu_integral_image.cu
    ${SRC}/cu_convolution.cu
    ${SRC}/cu_deconvolution.cu
    ${SRC}/cu_rof_denoising.cu
    ${SRC}/cu_tgv.cu
    ${SRC}/cu_segment_test.cu
    ${SRC}/cu_painting.cu
    ${SRC}/cu_remap.cu
    ${SRC}/cu_raycast.cu
    ${SRC}/cu_sdffusion.cu
)

################################################################################
# Find required dependencies

# Find CUDA toolkit. Thrust and NPP are bundled with CUDA toolkit.
find_package(CUDA 7.0 REQUIRED)
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
list(APPEND LINK_LIBS ${CUDA_npp_LIBRARY} ${CUDA_LIBRARIES} )

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

#######################################################
## Create configure file for inclusion in library

CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/kangaroo/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/kangaroo/config.h"
)

################################################################################
# Include collected includes / libraries
include_directories( ${USER_INC} )
include_directories( ${LIB_INC_DIR} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR}/include )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include/kangaroo )
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

# Version information
#configure_file("${PROJECT_NAME}ConfigVersion.cmake.in"
#  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

#######################################################
## Install it all.

install_package(
    PKG_NAME ${PROJECT_NAME}
    LIB_NAME ${LIBRARY_NAME}
    VERSION ${VERSION}
    DESCRIPTION "CUDA helper library for vision."
    INSTALL_INCLUDE_DIR true
    INSTALL_GENERATED_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/include/kangaroo/config.h"
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    INCLUDE_DIRS ${USER_INC} ${LIB_INC_DIR}
    LINK_LIBS ${LINK_LIBS}
    LINK_DIRS "${CMAKE_INSTALL_PREFIX}/lib"
    )

#set( KANGAROO_DIR ${CMAKE_CURRENT_BINARY_DIR} )
#set( KANGAROO_LIBRARIES ${REQUIRED_LIBRARIES} kangaroo )
#set( KANGAROO_INCLUDE_DIRS ${REQUIRED_INCLUDE_DIRS} )
link_libraries( ${REQUIRED_LIBRARIES} kangaroo )

if(BUILD_APPLICATIONS)
  add_subdirectory(applications)
endif()

# make an uninstall target
include(${CMAKE_MODULE_PATH}/cmake_uninstall.cmake.in)
add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

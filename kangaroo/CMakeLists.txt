find_package(CUDA REQUIRED)

set(CUDA_PROPAGATE_HOST_FLAGS off)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Kepler OSX laptop
  set(CUDA_NVCC_FLAGS "-arch=sm_30 -use_fast_math" )
else()
  set(CUDA_NVCC_FLAGS "--pre-include /usr/local/cuda/include/undef_cuda.h -arch=sm_20 -use_fast_math" )
endif()

if(NOT CUDA_npp_LIBRARY)
    # Hack for old FindCUDA to locate libnpp.so
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/libnpp.dylib")
    else()
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnpp.so")
    endif()
endif()

list(APPEND SRC_H
    kangaroo.h
    Image.h Pyramid.h Volume.h BoundedVolume.h BoundingBox.h Sdf.h
    ImageApron.h sampling.h pixel_convert.h patch_score.h
    Mat.h MatUtils.h reweighting.h reduce.h InvalidValue.h
    CudaTimer.h LeastSquareSum.h hamming_distance.h
    variational.h Divergence.h disparity.h
    hamming_distance.h
)

list(APPEND SRC_CU
    cu_anaglyph.cu cu_bilateral.cu cu_convert.cu cu_index_buffer.cu
    cu_model_refinement.cu cu_dense_stereo.cu cu_lookup_warp.cu
    cu_median.cu cu_normals.cu cu_depth_tools.cu cu_plane_fit.cu
    cu_resample.cu cu_heightmap.cu cu_blur.cu cu_operations.cu
    cu_census.cu cu_semi_global_matching.cu
    cu_manhatten.cu cu_integral_image.cu
    cu_convolution.cu cu_deconvolution.cu
    cu_rof_denoising.cu cu_tgv.cu
    cu_segment_test.cu
    cu_painting.cu cu_remap.cu
    cu_raycast.cu cu_sdffusion.cu
)

find_package( Eigen3 REQUIRED )
find_package( ASSIMP REQUIRED )
if(EIGEN3_FOUND AND ASSIMP_FOUND)
    message(STATUS "eigen: ${EIGEN3_INCLUDE_DIR}")
    include_directories( ${EIGEN3_INCLUDE_DIR} )  
    include_directories( ${ASSIMP_INCLUDE_DIR} )  
    link_libraries( ${ASSIMP_LIBRARIES} )
    list(APPEND SRC_CU MarchingCubes.cpp )
endif()

cuda_add_library( ${LIBRARY_NAME} ${SRC_H} ${SRC_CU} )

##############################################################################
# Get full library name
GET_TARGET_PROPERTY( FULL_LIBRARY_NAME ${LIBRARY_NAME} LOCATION )
set( Kangaroo_LIBRARIES "${FULL_LIBRARY_NAME};${CUDA_LIBRARIES};${CUDA_npp_LIBRARY}" )
set( Kangaroo_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..;${PROJECT_BINARY_DIR};${CUDA_INCLUDE_DIRS}")

##############################################################################
# Export config file for easy inclusion into other projects: FIND_PACKAGE( Pangolin )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/KangarooConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/KangarooConfig.cmake @ONLY IMMEDIATE )
EXPORT( PACKAGE kangaroo )

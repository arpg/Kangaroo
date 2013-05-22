cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/")

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-std=c++11 -g")
endif()

find_package( Boost COMPONENTS thread REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )
link_libraries( ${Boost_LIBRARIES} )

# git clone git://github.com/stevenlovegrove/Pangolin.git
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
link_directories(${Pangolin_LIBRARY_DIRS})
link_libraries(${Pangolin_LIBRARIES})

# git clone git://github.com/stevenlovegrove/Sophus.git
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})
link_directories(${Sophus_LIBRARY_DIRS})
link_libraries(${Sophus_LIBRARIES})

# git clone http://robotics.gwu.edu/git/SceneGraph
find_package(SceneGraph REQUIRED)
include_directories(${SceneGraph_INCLUDE_DIRS})
link_libraries(${SceneGraph_LIBRARIES})

find_package( Kangaroo REQUIRED)
include_directories( ${Kangaroo_INCLUDE_DIRS} )
link_libraries(${Kangaroo_LIBRARIES})

add_executable( Anaglyph Anaglyph.cpp)
add_executable( BilateralFilter BilateralFilter.cpp)
add_executable( GuidedFilter GuidedFilter.cpp)
add_executable( Denoising Denoising.cpp)
add_executable( Deconvolution Deconvolution.cpp)
add_executable( Inpainting Inpainting.cpp)
add_executable( Features Features.cpp)
add_executable( Raycast Raycast.cpp)
add_executable( SdfFusion SdfFusion.cpp)
add_executable( SdfDifference SdfDifference.cpp)

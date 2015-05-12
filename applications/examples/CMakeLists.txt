cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/")

if( NOT MSVC )
    set( CMAKE_CXX_FLAGS "-std=c++0x -Wall ${CMAKE_CXX_FLAGS}" )
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif()
endif()

find_package( Kangaroo 0.1 REQUIRED)
include_directories( ${Kangaroo_INCLUDE_DIRS} )
link_libraries(${Kangaroo_LIBRARIES})

# git clone git://github.com/stevenlovegrove/Pangolin.git
find_package(Pangolin 0.1 QUIET)
if(Pangolin_FOUND)
    include_directories(${Pangolin_INCLUDE_DIRS})
    link_libraries(${Pangolin_LIBRARIES})
        
    add_executable( Anaglyph Anaglyph.cpp)
    add_executable( BilateralFilter BilateralFilter.cpp)
    add_executable( GuidedFilter GuidedFilter.cpp)
    add_executable( Denoising Denoising.cpp)
    add_executable( Deconvolution Deconvolution.cpp)
    add_executable( Inpainting Inpainting.cpp)
    add_executable( Features Features.cpp)

    # git clone git://github.com/stevenlovegrove/Sophus.git
    find_package(Sophus QUIET)

    # git clone http://robotics.gwu.edu/git/SceneGraph
    find_package(SceneGraph 0.1 QUIET)
    
    if(Sophus_FOUND AND SceneGraph_FOUND)
        include_directories(${Sophus_INCLUDE_DIRS})
        link_libraries(${Sophus_LIBRARIES})
        
        include_directories(${SceneGraph_INCLUDE_DIRS})
        link_libraries(${SceneGraph_LIBRARIES})
    
        add_executable( Raycast Raycast.cpp)
        add_executable( SdfFusion SdfFusion.cpp)
        add_executable( SdfDifference SdfDifference.cpp)
    endif()
endif()

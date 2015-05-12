cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -g")

link_libraries(${Kangaroo_LIBRARIES})

# git clone git://github.com/arpg/Pangolin.git
find_package(Pangolin 0.1 QUIET)

# git clone git://github.com/arpg/Sophus.git
find_package(Sophus QUIET)

# git clone git://github.com/arpg/SceneGraph
find_package(SceneGraph 0.1 QUIET)

# git clone git://github.com/arpg/calibu
find_package(Calibu 0.1 QUIET)

# git clone git://github.com/arpg/HAL
find_package(HAL 0.1 QUIET)

if(Pangolin_FOUND AND Sophus_FOUND AND SceneGraph_FOUND AND Calibu_FOUND AND HAL_FOUND )
    include_directories(${Pangolin_INCLUDE_DIRS})
    link_libraries(${Pangolin_LIBRARIES})

    include_directories(${Sophus_INCLUDE_DIR})

    include_directories(${SceneGraph_INCLUDE_DIRS})
    link_libraries(${SceneGraph_LIBRARIES})

    include_directories(${Calibu_INCLUDE_DIRS})
    link_libraries(${Calibu_LIBRARIES})

    include_directories(${HAL_INCLUDE_DIRS})
    link_libraries(${HAL_LIBRARIES})

    add_executable(SimpleStereo main.cpp)
endif()

cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules/")

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-std=c++11 -g")
endif()

find_package( Boost COMPONENTS thread REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )
link_libraries( ${Boost_LIBRARIES} )

# git clone git://github.com/stevenlovegrove/Pangolin.git
find_package(Pangolin 0.1 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
link_directories(${Pangolin_LIBRARY_DIRS})
link_libraries(${Pangolin_LIBRARIES})

# git clone git://github.com/stevenlovegrove/Sophus.git
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})
link_directories(${Sophus_LIBRARY_DIRS})
link_libraries(${Sophus_LIBRARIES})

# git clone http://robotics.gwu.edu/git/SceneGraph
find_package(SceneGraph 0.1 REQUIRED)
include_directories(${SceneGraph_INCLUDE_DIRS})
link_libraries(${SceneGraph_LIBRARIES})

# git clone http://robotics.gwu.edu/git/RPG
find_package( HAL 0.1 REQUIRED )
include_directories( ${HAL_INCLUDE_DIRS} )
link_libraries(${HAL_LIBRARIES})

find_package( Calibu 0.1 REQUIRED )
include_directories( ${Calibu_INCLUDE_DIRS} )
link_directories(${Calibu_LIBRARY_DIRS})
link_libraries(${Calibu_LIBRARIES})

find_package( Kangaroo REQUIRED)
include_directories( ${Kangaroo_INCLUDE_DIRS} )
link_libraries(${Kangaroo_LIBRARIES})

add_executable( KinectFusion main.cpp)

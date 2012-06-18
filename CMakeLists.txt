project( Kangaroo )
cmake_minimum_required(VERSION 2.8)

find_package( SimpleGui REQUIRED )
include_directories( ${SimpleGui_INCLUDE_DIRS} )

find_package( RPG REQUIRED )
include_directories( ${RPG_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )

# git://github.com/stevenlovegrove/fiducials.git
find_package( Fiducials REQUIRED )
include_directories( ${Fiducials_INCLUDE_DIRS} )
link_directories(${Fiducials_LIBRARY_DIRS})

# git://github.com/stevenlovegrove/Sophus.git
find_package(Sophus REQUIRED)
include_directories( ${Sophus_INCLUDE_DIR} )
link_directories( ${Sophus_LIBRARY_DIRS} )

# https://github.com/obiou/FPL.git
FIND_PACKAGE(CCAMERAMODEL QUIET)
include_directories( ${CCAMERAMODEL_INCLUDE_DIR} )

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "-std=c++0x")
ENDIF()

set( BuildImuTest true CACHE "" force )
if( BuildImuTest )
    add_executable( ImuTest ImuTest.cpp )
    target_link_libraries( ImuTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
endif()

set( BuildMeshTest true CACHE "" force )
if( BuildMeshTest )
    add_executable( MeshTest MeshTest.cpp )
    target_link_libraries( MeshTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
endif()

set( BuildTriggerTest true CACHE "" force )
if( BuildTriggerTest )
    add_executable( TriggerTest TriggerTest.cpp )
    target_link_libraries( TriggerTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
endif()

set( BuildStereoCalibration true CACHE "" force )
if( BuildStereoCalibration )
    add_executable( StereoCalibration StereoCalibration.cpp )
    target_link_libraries( StereoCalibration ${CCAMERAMODEL_LIBRARIES} ${RPG_LIBRARIES} ${SimpleGui_LIBRARIES} ${Fiducials_LIBRARIES} ${Sophus_LIBRARIES} ${OpenCV_LIBS} )
endif()

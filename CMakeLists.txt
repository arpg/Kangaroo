project( Kangaroo )
cmake_minimum_required(VERSION 2.8)

find_package( SimpleGui REQUIRED )
include_directories( ${SimpleGui_INCLUDE_DIRS} )

find_package( RPG REQUIRED )
include_directories( ${RPG_INCLUDE_DIRS} )

set( BuildImuTest true CACHE "" force )
if( BuildImuTest )
    add_executable( ImuTest ImuTest.cpp )
    target_link_libraries( ImuTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
    add_dependencies( ImuTest SimpleGui RPG )
endif()

set( BuildMeshTest true CACHE "" force )
if( BuildMeshTest )
    add_executable( MeshTest MeshTest.cpp )
    target_link_libraries( MeshTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
    add_dependencies( MeshTest SimpleGui )
endif()

set( BuildCamTest true CACHE "" force )
if( BuildCamTest )
    add_executable( CamTest CamTest.cpp )
    target_link_libraries( CamTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
    add_dependencies( CamTest SimpleGui )
endif()

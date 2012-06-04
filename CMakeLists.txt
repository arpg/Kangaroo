project( Kangaroo )
cmake_minimum_required(VERSION 2.8)

find_package( SimpleGui REQUIRED )
include_directories( ${SimpleGui_INCLUDE_DIRS} )

find_package( RPG REQUIRED )
include_directories( ${RPG_INCLUDE_DIRS} )

set( BuildMeshTest true CACHE "" force )
if( BuildMeshTest )
    add_executable( MeshTest MeshTest.cpp )
    target_link_libraries( MeshTest ${SimpleGui_LIBRARIES} ${RPG_LIBRARIES})
    add_dependencies( MeshTest SimpleGui )
endif()

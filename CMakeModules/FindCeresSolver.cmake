# - Try to find libCeresSolver
#
#  CeresSolver_FOUND - system has libCeresSolver
#  CeresSolver_INCLUDE_DIRS - the libCeresSolver include directories
#  CeresSolver_LIBRARIES - link these to use libCeresSolver

FIND_PATH(
  CeresSolver_INCLUDE_DIRS
  NAMES ceres/ceres.h
  PATHS /usr/include /usr/local/include /opt/local/include
)

FIND_LIBRARY(
  CeresSolver_LIBRARY
  NAMES ceres
  PATHS /usr/lib /usr/local/lib
)

FIND_LIBRARY(
  GLOG_LIBRARY
  NAMES glog
  PATHS /usr/lib /usr/local/lib
)

IF(CeresSolver_INCLUDE_DIRS AND CeresSolver_LIBRARY AND GLOG_LIBRARY)
   SET(CeresSolver_LIBRARIES "${CeresSolver_LIBRARY};${GLOG_LIBRARY}")
   SET(CeresSolver_FOUND TRUE)
ENDIF()

FIND_PACKAGE(Protobuf)
IF(${PROTOBUF_FOUND})
  LIST(APPEND CeresSolver_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
  LIST(APPEND CeresSolver_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/internal)
endif()


IF(CeresSolver_FOUND)
   IF(NOT CeresSolver_FIND_QUIETLY)
      MESSAGE(STATUS "Found CeresSolver: ${CeresSolver_LIBRARIES}")
   ENDIF(NOT CeresSolver_FIND_QUIETLY)
ELSE(CeresSolver_FOUND)
   IF(CeresSolver_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find CeresSolver")
   ENDIF(CeresSolver_FIND_REQUIRED)
ENDIF(CeresSolver_FOUND)

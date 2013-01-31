# Ikaros-project.org
#
# Find the Phidgets includes and library
#
# This module defines
# PHIDGETS_INCLUDE_DIR
# PHIDGETS_LIBRARIES
# PHIDGETS_FOUND

# Checking OS
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

find_library(PHIDGETS_LIBRARY Phidget21)

        find_path(PHIDGETS_INCLUDE_DIR
    NAMES
      phidget21.h
    PATHS
                /usr/local/include
                /usr/include
        )

if (PHIDGETS_LIBRARY)
	set(PHIDGETS_LIBRARIES
	${PHIDGETS_LIBRARY}
	)

endif (PHIDGETS_LIBRARY)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	find_path(PHIDGETS_INCLUDE_DIR
    NAMES
      phidget21.h
    PATHS
		/usr/local/include
		/usr/include
  	)
  	
	find_library(PHIDGETS_LIBRARIES
    NAMES
    	phidget21
    PATHS
    	/usr/local/lib
    	/usr/lib
	)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if (PHIDGETS_LIBRARIES)
	message(STATUS "Found Phidget:")
	message(STATUS " - Includes: ${PHIDGETS_INCLUDE_DIR}")
	message(STATUS " - Libraries: ${PHIDGETS_LIBRARIES}")
	set(PHIDGETS_FOUND "YES" )
endif (PHIDGETS_LIBRARIES)


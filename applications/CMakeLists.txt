# Reset flag from options set for library building (in particular stdlibc++).
set(CMAKE_CXX_FLAGS "")

ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(kinectfusion)
ADD_SUBDIRECTORY(stereo)

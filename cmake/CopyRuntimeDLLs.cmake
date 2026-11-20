# Copy all runtime DLLs from SOURCE_DIR to DEST_DIR.
#
# Used on Windows to make test and example executables locate the shared
# weif library and its runtime dependencies (for example FFTW) in the
# build tree. CMake does not copy DLLs of project targets automatically.
#
# Usage:
#   cmake -DSOURCE_DIR=<dir> -DDEST_DIR=<dir> -P CopyRuntimeDLLs.cmake

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED DEST_DIR)
	message(FATAL_ERROR "SOURCE_DIR and DEST_DIR must be defined")
endif()

file(GLOB _runtime_dlls "${SOURCE_DIR}/*.dll")
if(_runtime_dlls)
	file(COPY ${_runtime_dlls} DESTINATION "${DEST_DIR}")
endif()

find_package(Boost REQUIRED)
find_package(xtensor 0.26 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Rapidcsv REQUIRED)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(FFTW3L REQUIRED IMPORTED_TARGET fftw3l)

check_boost_math_special_functions(HAVE_CORRECT_SPECIAL_FUNCTIONS TARGET Boost::boost)
if(NOT HAVE_CORRECT_SPECIAL_FUNCTIONS)
	message(FATAL_ERROR "Boost::boost special functions are defective in your version")
endif()

file(GLOB_RECURSE SOURCES *.cpp)
add_library(weif ${SOURCES})
target_compile_options(weif
	PUBLIC "-Wall;-ffast-math;-fno-finite-math-only"
	PUBLIC "$<$<CONFIG:Release>:-O3>"
	PUBLIC "$<$<CONFIG:RelWithDebInfo>:-O3>")
target_compile_definitions(weif
	PUBLIC "$<$<CONFIG:Release>:NDEBUG>"
	PUBLIC "$<$<CONFIG:RelWithDebInfo>:NDEBUG>")
target_include_directories(weif
	PUBLIC "${PROJECT_SOURCE_DIR}/include"
	PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(weif Boost::boost PkgConfig::FFTW3F PkgConfig::FFTW3 PkgConfig::FFTW3L Rapidcsv::Rapidcsv)
#
# Here are a set of rules to help you update your library version information:
#
#    If the library source code has changed at all since the last update,
#        then increment revision (‘c:r:a’ becomes ‘c:r+1:a’).
#    If any interfaces have been added, removed, or changed since the last update,
#        increment current, and set revision to 0.
#    If any interfaces have been added since the last public release,
#        then increment age.
#    If any interfaces have been removed or changed since the last public release,
#        then set age to 0.
#
# set_target_properties(weif PROPERTIES VERSION c-a.a.r SOVERSION c-a)
#
set_target_properties(weif PROPERTIES VERSION 1.0.0 SOVERSION 1)
set_target_properties(weif PROPERTIES CXX_VISIBILITY_PRESET hidden)

generate_export_header(weif)

install(TARGETS weif
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/weif_export.h"
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

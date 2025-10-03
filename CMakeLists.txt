cmake_minimum_required (VERSION 3.10)
project(libweif LANGUAGES C CXX VERSION 0.1.0)

OPTION(BUILD_EXAMPLE "libweif example" ON)
OPTION(BUILD_TEST    "libweif test"    ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 17)

include(CheckCXXSourceRuns)
include(CheckIncludeFileCXX)
include(GNUInstallDirs)
include(GenerateExportHeader)
include(CheckIPOSupported)

check_ipo_supported(RESULT HAS_LTO_SUPPORT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ffast-math -fno-finite-math-only")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG")

find_package(Boost COMPONENTS
	program_options REQUIRED)
find_package(Doxygen REQUIRED)
find_package(xtensor REQUIRED VERSION 0.26)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(FFTW3L REQUIRED IMPORTED_TARGET fftw3l)
CHECK_INCLUDE_FILE_CXX("rapidcsv.h" RAPIDCSV_FOUND)
if(NOT RAPIDCSV_FOUND)
	message(FATAL_ERROR "rapidcsv.h not found")
endif()

check_cxx_source_runs("
#include <limits>
#include <boost/math/special_functions/bessel.hpp>

int main() {
	using namespace boost::math;
	constexpr auto inf = std::numeric_limits<double>::infinity();
	return !(cyl_bessel_j(1, inf) == 0.0 && sinc_pi(inf) == 0.0);
}
" HAVE_CORRECT_SPECIAL_FUNCTIONS)

if(NOT HAVE_CORRECT_SPECIAL_FUNCTIONS)
	message(FATAL_ERROR "boost::math special functions are defective in your version")
endif()

include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("${CMAKE_CURRENT_BINARY_DIR}")

file(GLOB_RECURSE SOURCES src/*.cpp)
add_library(weif SHARED ${SOURCES})
target_link_libraries(weif Boost::boost PkgConfig::FFTW3F PkgConfig::FFTW3 PkgConfig::FFTW3L)
if(HAS_LTO_SUPPORT)
	set_property(TARGET weif PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif(HAS_LTO_SUPPORT)
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

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXCLUDE_SYMBOLS detail)
doxygen_add_docs(docs ${SOURCES} include)

if(BUILD_TEST)
pkg_check_modules(CPPUNIT REQUIRED IMPORTED_TARGET cppunit)
enable_testing()
file(GLOB_RECURSE TESTS test/*.cpp)
foreach(test_source IN ITEMS ${TESTS})
	string(REPLACE ${CMAKE_SOURCE_DIR}/ "" test_name ${test_source})
	string(REPLACE / _ test_name ${test_name})
	string(REPLACE .cpp "" test_name ${test_name})
	add_executable(${test_name} ${test_source})
	target_link_libraries(${test_name} weif PkgConfig::CPPUNIT)
	if(HAS_LTO_SUPPORT)
		set_property(TARGET ${test_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
	endif(HAS_LTO_SUPPORT)
	add_test(${test_name} ${test_name})
endforeach(test_source)
endif()

if(BUILD_EXAMPLE)
file(GLOB_RECURSE EXAMPLES example/*.cpp)
foreach(example_source IN ITEMS ${EXAMPLES})
	string(REPLACE ${CMAKE_SOURCE_DIR}/ "" example_name ${example_source})
	string(REPLACE / _ example_name ${example_name})
	string(REPLACE .cpp "" example_name ${example_name})
	add_executable(${example_name} ${example_source})
	target_link_libraries(${example_name} weif Boost::boost Boost::program_options)
	if(HAS_LTO_SUPPORT)
		set_property(TARGET ${example_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
	endif(HAS_LTO_SUPPORT)
endforeach(example_source)
endif()

install(TARGETS weif
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/weif
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	FILES_MATCHING PATTERN "*.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/weif_export.h"
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
configure_file("weif.pc.in" "weif.pc" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/weif.pc"
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

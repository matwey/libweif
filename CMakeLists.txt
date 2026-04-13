cmake_minimum_required (VERSION 3.10)
project(libweif LANGUAGES C CXX VERSION 0.2.3)

OPTION(BUILD_WEIF     "libweif"               ON)
OPTION(BUILD_DOCS     "libweif documentation" OFF)
OPTION(BUILD_EXAMPLES "libweif example"       ON)
OPTION(BUILD_TESTS    "libweif test"          ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 20)

include(GNUInstallDirs)
include(GenerateExportHeader)
include(CheckIPOSupported)
include(CheckBoostMathSpecialFunctions)

check_ipo_supported(RESULT HAS_LTO_SUPPORT)
if(HAS_LTO_SUPPORT)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif(HAS_LTO_SUPPORT)

if (BUILD_WEIF)
	add_subdirectory("src")

	install(DIRECTORY include/weif
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
		FILES_MATCHING PATTERN "*.h")
	configure_file("weif.pc.in" "weif.pc" @ONLY)
	install(FILES "${CMAKE_CURRENT_BINARY_DIR}/weif.pc"
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
endif()

if (BUILD_DOCS)
	add_subdirectory("docs")
endif()

if (BUILD_EXAMPLES)
	add_subdirectory("examples")
endif()

if (BUILD_TESTS)
	enable_testing()

	add_subdirectory("tests")
endif()

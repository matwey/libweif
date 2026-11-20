find_package(PkgConfig REQUIRED)
pkg_check_modules(CPPUNIT REQUIRED IMPORTED_TARGET cppunit)

get_target_property(WEIF_TARGET_TYPE weif TYPE)

file(GLOB_RECURSE TESTS *.cpp)
foreach(test_source IN ITEMS ${TESTS})
	string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/ "" test_name ${test_source})
	string(REPLACE / _ test_name ${test_name})
	string(REPLACE .cpp "" test_name ${test_name})
	string(PREPEND test_name "test_")
	add_executable(${test_name} ${test_source})
	target_link_libraries(${test_name} weif PkgConfig::CPPUNIT)
	add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
	if(WIN32 AND WEIF_TARGET_TYPE STREQUAL "SHARED_LIBRARY")
		add_custom_command(TARGET ${test_name} POST_BUILD
			COMMAND ${CMAKE_COMMAND}
				-DSOURCE_DIR=$<TARGET_FILE_DIR:weif>
				-DDEST_DIR=$<TARGET_FILE_DIR:${test_name}>
				-P "${PROJECT_SOURCE_DIR}/cmake/CopyRuntimeDLLs.cmake")
	endif()
endforeach(test_source)

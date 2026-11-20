find_package(Boost COMPONENTS program_options REQUIRED)

get_target_property(WEIF_TARGET_TYPE weif TYPE)

file(GLOB_RECURSE EXAMPLES *.cpp)
foreach(example_source IN ITEMS ${EXAMPLES})
	string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/ "" example_name ${example_source})
	string(REPLACE / _ example_name ${example_name})
	string(REPLACE .cpp "" example_name ${example_name})
	add_executable(${example_name} ${example_source})
	target_link_libraries(${example_name} weif Boost::program_options)
	if(WIN32 AND WEIF_TARGET_TYPE STREQUAL "SHARED_LIBRARY")
		add_custom_command(TARGET ${example_name} POST_BUILD
			COMMAND ${CMAKE_COMMAND}
				-DSOURCE_DIR=$<TARGET_FILE_DIR:weif>
				-DDEST_DIR=$<TARGET_FILE_DIR:${example_name}>
				-P "${PROJECT_SOURCE_DIR}/cmake/CopyRuntimeDLLs.cmake")
	endif()
endforeach(example_source)

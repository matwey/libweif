#[=======================================================================[.rst:
FindRapidcsv
------------

Finds the Rapidcsv header-only library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported target, if found:

``Rapidcsv::Rapidcsv``
  The Rapidcsv header-only library target

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Rapidcsv_FOUND``
  True if the system has the Rapidcsv library.
``Rapidcsv_INCLUDE_DIRS``
  Include directories needed to use Rapidcsv.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Rapidcsv_INCLUDE_DIR``
  The directory containing ``rapidcsv.h``.
``HAVE_Rapidcsv_HEADER``
  Boolean indicating whether the rapidcsv header compiles successfully.

Examples
^^^^^^^^

.. code-block:: cmake

    find_package(Rapidcsv REQUIRED)

    add_executable(my_app main.cpp)
    target_link_libraries(my_app PRIVATE Rapidcsv::Rapidcsv)

#]=======================================================================]

find_path(Rapidcsv_INCLUDE_DIR NAMES rapidcsv.h)

if(Rapidcsv_INCLUDE_DIR)
	include(CheckIncludeFileCXX)

	set(CMAKE_REQUIRED_INCLUDES_SAVE ${CMAKE_REQUIRED_INCLUDES})
	set(CMAKE_REQUIRED_QUIET_SAVE ${CMAKE_REQUIRED_QUIET})

	list(APPEND CMAKE_REQUIRED_INCLUDES ${Rapidcsv_INCLUDE_DIR})
	set(CMAKE_REQUIRED_QUIET TRUE)

	check_include_file_cxx("rapidcsv.h" HAVE_Rapidcsv_HEADER)

	set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES_SAVE})
	set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAVE})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Rapidcsv
	FOUND_VAR Rapidcsv_FOUND
	REQUIRED_VARS HAVE_Rapidcsv_HEADER Rapidcsv_INCLUDE_DIR)

if(Rapidcsv_FOUND AND NOT TARGET Rapidcsv::Rapidcsv)
	add_library(Rapidcsv::Rapidcsv INTERFACE IMPORTED)

	set_target_properties(Rapidcsv::Rapidcsv PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${Rapidcsv_INCLUDE_DIR}")
endif()

if(Rapidcsv_FOUND)
	set(Rapidcsv_INCLUDE_DIRS ${Rapidcsv_INCLUDE_DIR})
endif()

mark_as_advanced(Rapidcsv_INCLUDE_DIR HAVE_Rapidcsv_HEADER)

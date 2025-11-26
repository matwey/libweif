#[=======================================================================[.rst:
CheckBoostMathSpecialFunctions
------------------------------

Checks if boost::math special functions work correctly.

.. command:: check_boost_math_special_functions

  ::

    check_boost_math_special_functions(<result_var> [TARGET <boost_target>])

  Checks whether boost::math special functions (cyl_bessel_j and sinc_pi)
  behave correctly with infinity values.

  ``<result_var>``
    Variable to store the result

  ``TARGET <boost_target>``
    Optional Boost target to use for linking (default: Boost::boost)

#]=======================================================================]
function(check_boost_math_special_functions RESULT_VAR)
	set(options "")
	set(oneValueArgs TARGET)
	set(multiValueArgs "")
	cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

	include(CheckCXXSourceRuns)

	if(NOT ARG_TARGET)
		set(BOOST_TARGET "Boost::boost")
	else()
		set(BOOST_TARGET "${ARG_TARGET}")
	endif()

	set(CMAKE_REQUIRED_LIBRARIES_SAVE ${CMAKE_REQUIRED_LIBRARIES})
	set(CMAKE_REQUIRED_QUIET_SAVE ${CMAKE_REQUIRED_QUIET})

	list(APPEND CMAKE_REQUIRED_LIBRARIES ${BOOST_TARGET})
	set(CMAKE_REQUIRED_QUIET TRUE)

	check_cxx_source_runs("
#include <limits>
#include <boost/math/special_functions/bessel.hpp>

int main() {
using namespace boost::math;
constexpr auto inf = std::numeric_limits<double>::infinity();
return !(cyl_bessel_j(1, inf) == 0.0 && sinc_pi(inf) == 0.0);
}
" HAVE_CORRECT_SPECIAL_FUNCTIONS_INTERNAL)

	set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAVE})
	set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_SAVE})

	set(${RESULT_VAR} ${HAVE_CORRECT_SPECIAL_FUNCTIONS_INTERNAL} PARENT_SCOPE)
endfunction()

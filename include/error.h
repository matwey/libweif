#ifndef _WEIF_ERROR_H
#define _WEIF_ERROR_H

#include <sstream>
#include <stdexcept>
#include <string>

#include <gsl/gsl_errno.h>

#include <weif_export.h>

namespace weif {

struct WEIF_EXPORT error:
	public std::runtime_error {

	error(const std::string& what);
	error(const char* what);
};

struct WEIF_EXPORT non_uniform_grid:
	public error {

	template<class T>
	non_uniform_grid(std::size_t position, T actual, T expected):
		error(reinterpret_cast<std::ostringstream&>(std::ostringstream() << "Non uniform input grid at position "
			<< position << ", actual value "
			<< actual << ", expected "
			<< expected).str()) {}
};

struct WEIF_EXPORT mismatched_grids:
	public error {

	mismatched_grids() noexcept;
};

struct WEIF_EXPORT gsl_error:
	public error {

	gsl_error(const int gsl_errno) noexcept;
};

} // weif

#endif // _WEIF_ERROR_H

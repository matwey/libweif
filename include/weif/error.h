/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_ERROR_H
#define _WEIF_ERROR_H

#include <sstream>
#include <stdexcept>
#include <string>

#include <weif_export.h>

namespace weif {

/**
 * @brief Base exception class for weif library errors
 *
 * Inherits from std::runtime_error to provide standard exception handling.
 * Used as base class for all weif-specific exceptions.
 */
struct WEIF_EXPORT error:
	public std::runtime_error {

	/**
	 * @brief Construct with string message
	 * @param what Error description
	 */
	error(const std::string& what);

	/**
	 * @brief Construct with C-string message
	 * @param what Error description
	 */
	error(const char* what);
};

/**
 * @brief Exception thrown for non-uniform grid input
 *
 * Indicates that input data expected to form a uniform grid
 * contains inconsistent spacing between values.
 *
 * @see uniform_grid
 */
struct WEIF_EXPORT non_uniform_grid:
	public error {

	/**
	 * @brief Construct with position and value details
	 * @tparam T Numeric type of the grid values
	 * @param position Index where non-uniformity was detected
	 * @param actual Value found at the position
	 * @param expected Value expected for uniform grid
	 *
	 * Formats an error message showing:
	 * - Position of the inconsistency
	 * - Actual value encountered
	 * - Expected uniform value
	 */
	template<class T>
	non_uniform_grid(std::size_t position, T actual, T expected):
		error(reinterpret_cast<std::ostringstream&>(std::ostringstream() << "Non uniform input grid at position "
			<< position << ", actual value "
			<< actual << ", expected "
			<< expected).str()) {}
};

/**
 * @brief Exception thrown for incompatible grid operations
 *
 * Indicates that two grids cannot be combined or compared because they have:
 * - Different spacing
 * - Misaligned origins
 * - No overlapping range
 *
 * @see uniform_grid
 */
struct WEIF_EXPORT mismatched_grids:
	public error {

	/**
	 * @brief Construct with default message
	 *
	 * Creates exception indicating grids cannot be combined due to
	 * incompatible spacing or alignment.
	 */
	mismatched_grids() noexcept;
};

} // weif

#endif // _WEIF_ERROR_H

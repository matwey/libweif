/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_SQUARE_H
#define _WEIF_AF_SQUARE_H

#include <cmath>
#include <type_traits>

#include <xtensor/xmath.hpp>
#include <xtensor/xvectorize.hpp>

#include <weif/math.h>
#include <weif_export.h>


namespace weif {
namespace af {

/**
 * @brief Aperture filter function for a square aperture
 *
 * @tparam T Numeric type used for calculations.
 *
 * The aperture filter is defined in Cartesian coordinates as:
 * \f[
 * A(u_x, u_y) = \mathrm{sinc}^2(\pi u_x) \cdot \mathrm{sinc}^2(\pi u_y),
 * \f]
 * representing the Fourier transform of a square pupil function.
 */
template<class T>
struct WEIF_EXPORT square {
	using value_type = T; ///< Numeric type used for calculations

	/**
	 * @brief Call operator for square aperture filter in Cartesian coordinates
	 *
	 * Evaluates the squared 2D sinc function:
	 * \f[
	 * A(u_x, u_y) = \mathrm{sinc}^2(\pi u_x) \cdot \mathrm{sinc}^2(\pi u_y).
	 * \f]
	 *
	 * @param ux Dimensionless spatial frequency in x-direction
	 * @param uy Dimensionless spatial frequency in y-direction
	 * @return Aperture filter value at specified frequencies
	 */
	value_type operator() (value_type ux, value_type uy) const noexcept {
		using namespace std;
		using boost::math::sinc_pi;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(sinc_pi(ux * PI) * sinc_pi(uy * PI), 2);
	}

	/**
	 * @brief Vectorized evaluation for square aperture on frequency grids
	 *
	 * Computes the 2D sinc-squared function for grid inputs:
	 * \f[
	 * A(u_x, u_y) = \mathrm{sinc}^2(\pi u_x) \cdot \mathrm{sinc}^2(\pi u_y).
	 * \f]
	 *
	 * @param ex Tensor of x-component frequencies
	 * @param ey Tensor of y-component frequencies
	 * @return (Nx, Ny) shaped tensor of filter values
	 */
	template<class EX, class EY, xt::enable_xexpression<EX, bool> = true, xt::enable_xexpression<EY, bool> = true>
	auto operator() (EX&& ex, EY&& ey) const noexcept {
		using xvalue_type = std::common_type_t<xt::get_value_type_t<std::decay_t<EX>>, xt::get_value_type_t<std::decay_t<EY>>>;

		auto [xx, yy] = xt::meshgrid(
			math::sinc_pi(xt::numeric_constants<xvalue_type>::PI * std::forward<EX>(ex)),
			math::sinc_pi(xt::numeric_constants<xvalue_type>::PI * std::forward<EY>(ey)));

		return xt::square(std::move(xx) * std::move(yy));
	}
};

} // af
} // weif

#endif // _WEIF_AF_SQUARE_H

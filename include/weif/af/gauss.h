/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_GAUSS_H
#define _WEIF_AF_GAUSS_H

#include <cmath>

#include <xtensor/utils/xutils.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xvectorize.hpp>

#include <weif_export.h>


namespace weif {
namespace af {

/**
 * @brief Aperture filter for Gaussian aperture
 *
 * @tparam T Numeric type used for calculations.
 *
 * Represents a Gaussian aperture filter defined as:
 * \f[
 * A(u) = e^{-u^2}
 * \f]
 * in radial coordinates, and:
 * \f[
 * A(u_x, u_y) = e^{-(u_x^2 + u_y^2)}
 * \f]
 * in Cartesian coordinates. This corresponds to the Fourier transform of a Gaussian
 * pupil function, commonly used to model apodized optical systems.
 */
template<class T>
struct WEIF_EXPORT gauss {
	using value_type = T; ///< Numeric type used for calculations

	/**
	 * @brief Call operator for Gaussian aperture filter in radial coordinates
	 *
	 * Evaluates the Gaussian function:
	 * \f$ A(u) = e^{-u^2} \f$.
	 *
	 * @param u Dimensionless spatial frequency magnitude
	 * @return Filter transmission value at given frequency
	 */
	value_type operator() (value_type u) const noexcept {
		return std::exp(-u*u);
	}

	/**
	 * @brief Call operator for Gaussian aperture filter in Cartesian coordinates
	 *
	 * Evaluates the 2D Gaussian function:
	 * \f$ A(u_x, u_y) = e^{-(u_x^2 + u_y^2)} \f$
	 *
	 * @param ux Dimensionless spatial frequency in x-direction
	 * @param uy Dimensionless spatial frequency in y-direction
	 * @return Filter transmission value at given frequency coordinates
	 */
	value_type operator() (value_type ux, value_type uy) const noexcept {
		return std::exp(-ux*ux-uy*uy);
	}

        /**
         * @brief Vectorized evaluation for Gaussian aperture filter (radial coordinates)
         *
         * Evaluates Gaussian aperture filter for array of frequencies:
         * \f$ A(u) = e^{-u^2} \f$
         *
         * @param e Input tensor of frequency magnitudes
         * @return Tensor of transmission values with same shape as input
         */
	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		return xt::exp(-xt::square(std::forward<E>(e)));
	}

        /**
         * @brief Vectorized evaluation for Gaussian aperture filter (Cartesian coordinates)
         *
         * Evaluates 2D Gaussian aperture filter on a frequency grid:
         * \f$ A(u_x, u_y) = e^{-(u_x^2 + u_y^2)} \f$
         *
         * @param ex Tensor of x-component frequencies
         * @param ey Tensor of y-component frequencies
         * @return (Nx, Ny) shaped tensor of transmission values
         */
	template<class EX, class EY, xt::enable_xexpression<EX, bool> = true, xt::enable_xexpression<EY, bool> = true>
	auto operator() (EX&& ex, EY&& ey) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<EX>(ex), std::forward<EY>(ey));

		return xt::exp(-xt::square(std::move(xx)) - xt::square(std::move(yy)));
	}
};

} // af
} // weif

#endif // _WEIF_AF_GAUSS_H


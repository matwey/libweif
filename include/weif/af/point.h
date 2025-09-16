/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_POINT_H
#define _WEIF_AF_POINT_H

#include <weif_export.h>


namespace weif {
namespace af {

/**
 * @brief Aperture filter function for a point (infinitely small) aperture
 *
 * @tparam T Numeric type used for calculations
 *
 * The aperture filter is defined in both radial and Cartesian plane coordinates:
 * \f[
 * A(u) = A(u_x, u_y) = 1.
 * \f]
 * This represents an ideal point aperture.
 */
template<class T>
struct WEIF_EXPORT point {
	using value_type = T; ///< Numeric type used for calculations

        /**
         * @brief Call operator for point aperture filter in radial coordinates
         *
         * Evaluates the filter function for a given radial frequency:
         * \f$ A(u) = 1. \f$
         *
         * @param u Dimensionless spatial frequency magnitude (radial coordinate)
         * @return Aperture filter value (always 1 for point aperture)
         */
	value_type operator() (value_type u) const noexcept {
		return static_cast<value_type>(1);
	}

        /**
         * @brief Call operator for point aperture filter in Cartesian coordinates
         *
         * Evaluates the filter function for given frequency components:
         * \f$ A(u_x, u_y) = 1. \f$
         *
         * @param ux Dimensionless spatial frequency component in x-direction
         * @param uy Dimensionless spatial frequency component in y-direction
         * @return Aperture filter value (always 1 for point aperture)
         */
	value_type operator() (value_type ux, value_type uy) const noexcept {
		return static_cast<value_type>(1);
	}

        /**
         * @brief Call operator for point aperture filter with tensor input (radial coordinates)
         *
         * Evaluates the filter function for an array of frequency magnitudes:
         * \f$ A(u) = 1. \f$
         *
         * @param e Input tensor of dimensionless spatial frequency magnitudes
         * @return Tensor of aperture filter values (all ones) with same shape as input
         */
	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		return xt::ones_like(std::forward<E>(e));
	}

        /**
         * @brief Call operator for point aperture filter with tensor inputs (Cartesian coordinates)
         *
         * Evaluates the filter function on a grid of frequency components:
         * \f$ A(u_x, u_y) = \mathbf{1}. \f$
         *
         * @param ex Tensor of dimensionless x-component frequencies
         * @param ey Tensor of dimensionless y-component frequencies
         * @return (Nx, Ny) shaped tensor of aperture filter values (all ones)
         */
	template<class EX, class EY, xt::enable_xexpression<EX, bool> = true, xt::enable_xexpression<EY, bool> = true>
	auto operator() (EX&& ex, EY&& ey) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<EX>(ex), std::forward<EY>(ey));

		return xt::ones_like(std::move(xx));
	}
};

} // af
} // weif

#endif // _WEIF_AF_POINT_H

/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_WEIGHT_FUNCTION_2D_H
#define _WEIF_WEIGHT_FUNCTION_2D_H

#include <cmath>
#include <limits>
#include <utility>

#include <boost/math/quadrature/exp_sinh.hpp>

#include <xtensor/core/xexpression.hpp> // IWYU pragma: keep
#include <xtensor/core/xmath.hpp>

#include <weif/detail/weight_function_base.h>
#include <weif_export.h>


namespace weif {

/**
 * @brief Scintillation weight function for non axially symmetric power spectra
 *
 * @tparam T Numeric type used for calculations
 *
 * Computes the scintillation weight function used in atmospheric turbulence
 * analysis for non axially symmetric power spectra:
 * \f[
 * W(z) = 9.69 \cdot 10^{-3} \cdot 16 \pi^2 z^{5/6} \lambda^{-7/6} \int \mathbf{du} u^{-8/3} S(u) A\left(\frac{D}{\sqrt{\lambda z}} \mathbf{u}\right),
 * \f]
 * where \f$ S(u) \f$ is a spectral filter, \f$ \lambda \f$ is its equivalent wavelength, and \f$ A(\mathbf{u}) \f$ is an aperture filter.
 *
 * @par The library uses consistent units:
 * - Altitudes: kilometers (km)
 * - Wavelengths: nanometers (nm)
 * - Geometric scales: millimeters (mm)
 *
 * @see sf::poly::equiv_lambda()
 * @see weif::math::Kolmogorov_Cn2_scale
 */
template<class T>
class WEIF_EXPORT weight_function_2d:
	public detail::weight_function_base<T> {
public:
	using value_type = typename detail::weight_function_base<T>::value_type; ///< Numeric type for calculations

private:
	using function_type = std::function<value_type(value_type, value_type, value_type)>;

public:
	template<class SF, class AF>
	weight_function_2d(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, const uniform_grid<value_type>& grid):
		detail::weight_function_base<T>(lambda, aperture_scale, grid,
			detail::dimensionless_weight_function_2d(std::forward<SF>(spectral_filter), std::forward<AF>(aperture_filter), grid.values())) {}

	/**
	 * @brief Construct 2D weight function
	 * @param spectral_filter Spectral filter function
	 * @param lambda Wavelength in nanometers
	 * @param aperture_filter 2D aperture filter function
	 * @param aperture_scale Aperture scale in millimeters
	 * @param size Number of grid points for precomputation
	 *
	 * The weight function is precomputed on a grid of `size` nodes using
	 * numerical integration technique and subsequent interpolation is used
	 * when the weight_function_2d::operator()() is invoked.
	 *
	 * @see operator()()
	 */
	template<class SF, class AF>
	weight_function_2d(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, std::size_t size):
		weight_function_2d(std::forward<SF>(spectral_filter), lambda, std::forward<AF>(aperture_filter), aperture_scale,
			uniform_grid{static_cast<value_type>(0), static_cast<value_type>(1) / (size-1), size}) {}

	/**
	 * @brief Evaluate scintillation weight function at specific altitude
	 * @param altitude Atmospheric altitude in kilometers
	 * @return Weight value representing thin layer contribution to scintillation
	 */
	inline value_type operator() (value_type altitude) const noexcept {
		return detail::weight_function_base<T>::operator()(altitude);
	}

	/**
	 * @brief Evaluate scintillation weight function for tensor input
	 * @param e Altitude values expression in kilometers
	 * @return Tensor of scintillation weight function values
	 */
	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->operator()(x);
		}, e.derived_cast());
	}
};

extern template class weight_function_2d<float>;
extern template class weight_function_2d<double>;
extern template class weight_function_2d<long double>;

} // weif

#endif // _WEIF_WEIGHT_FUNCTION_2D_H

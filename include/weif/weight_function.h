/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_WEIGHT_FUNCTION_H
#define _WEIF_WEIGHT_FUNCTION_H

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include <boost/math/quadrature/exp_sinh.hpp>

#include <xtensor/xexpression.hpp> // IWYU pragma: keep
#include <xtensor/xmath.hpp>

#include <weif/detail/weight_function_base.h>
#include <weif_export.h>


namespace weif {

template<class T>
class WEIF_EXPORT weight_function:
	public detail::weight_function_base<T> {
public:
	using value_type = typename detail::weight_function_base<T>::value_type;

private:
	using function_type = std::function<value_type(value_type, value_type)>;

public:
	template<class SF, class AF>
	weight_function(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, const uniform_grid<value_type>& grid):
		detail::weight_function_base<T>(lambda, aperture_scale, grid,
			detail::dimensionless_weight_function(std::forward<SF>(spectral_filter), std::forward<AF>(aperture_filter), grid.values())) {}

	template<class SF, class AF>
	weight_function(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, std::size_t size):
		weight_function(std::forward<SF>(spectral_filter), lambda, std::forward<AF>(aperture_filter), aperture_scale,
			uniform_grid{static_cast<value_type>(0), static_cast<value_type>(1) / (size-1), size}) {}

	inline value_type operator() (value_type altitude) const noexcept {
		constexpr const auto PI = xt::numeric_constants<value_type>::PI;
		constexpr const value_type c = 2 * PI;

		return c * detail::weight_function_base<T>::operator()(altitude);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->operator()(x);
		}, e.derived_cast());
	}
};

extern template class weight_function<float>;
extern template class weight_function<double>;
extern template class weight_function<long double>;

} // weif

#endif // _WEIF_WEIGHT_FUNCTION_H

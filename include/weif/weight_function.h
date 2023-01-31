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

private:
	static auto integrate_weight_function(function_type&& fun, std::size_t size);

	weight_function(value_type lambda, value_type aperture_scale, function_type&& fun, std::size_t size):
		detail::weight_function_base<T>(lambda, aperture_scale, integrate_weight_function(std::forward<function_type>(fun), size)) {}

public:
	template<class SF, class AF>
	weight_function(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, std::size_t size):
		weight_function(lambda, aperture_scale,
			[spectral_filter = std::forward<SF>(spectral_filter), aperture_filter = std::forward<AF>(aperture_filter)] (value_type u, value_type x) noexcept -> value_type {
			using namespace std;

			if (u == static_cast<value_type>(0) || u == std::numeric_limits<value_type>::infinity())
				return static_cast<value_type>(0);

			if (u < static_cast<value_type>(1)) {
				return pow(u, static_cast<value_type>(4.0/3.0)) * spectral_filter.regular(u * u) * aperture_filter(x * u);
			}

			const auto t = pow(u, -static_cast<value_type>(8.0/3.0));
			if (t == static_cast<value_type>(0))
				return static_cast<value_type>(0);

			return t * spectral_filter(u * u) * aperture_filter(x * u);
		}, size) {}

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

template<class T>
auto weight_function<T>::integrate_weight_function(function_type&& fun, std::size_t size) {
	using boost::math::quadrature::exp_sinh;

	auto integrator = std::make_unique<exp_sinh<value_type>>();

	return xt::make_lambda_xfunction([integrator = std::move(integrator), fun = std::forward<function_type>(fun)] (value_type z) -> value_type {
		using namespace std::placeholders;

		if (z == static_cast<value_type>(0))
			return static_cast<value_type>(0);

		const auto tol = std::pow(std::numeric_limits<value_type>::epsilon(), static_cast<value_type>(2.0/3.0));
		const auto x = (static_cast<value_type>(1) - z) / z;

		return integrator->integrate(std::bind(std::cref(fun), _1, x), tol);
	}, xt::linspace(static_cast<value_type>(0), static_cast<value_type>(1), size));
}


extern template class weight_function<float>;
extern template class weight_function<double>;
extern template class weight_function<long double>;

} // weif

#endif // _WEIF_WEIGHT_FUNCTION_H

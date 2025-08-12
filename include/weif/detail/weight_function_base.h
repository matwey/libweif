/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_DETAIL_WEIGHT_FUNCTION_BASE_H
#define _WEIF_DETAIL_WEIGHT_FUNCTION_BASE_H

#include <cmath>
#include <functional>

#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/special_functions/sin_pi.hpp>
#include <boost/math/special_functions/cos_pi.hpp>

#include <xtensor/core/xexpression.hpp>
#include <xtensor/core/xmath.hpp>

#include <weif/detail/cubic_spline.h>
#include <weif/math.h>
#include <weif/uniform_grid.h>
#include <weif_export.h>


namespace weif {
namespace detail {

template<class T>
class WEIF_EXPORT weight_function_base {
public:
	using value_type = T;

private:
	value_type lambda_;
	value_type aperture_scale_;
	uniform_grid<value_type> grid_;
	cubic_spline<value_type> wf_;

protected:
	inline value_type operator() (value_type altitude) const noexcept {
		using namespace std;

		constexpr const auto PI = xt::numeric_constants<value_type>::PI;
		/* 1e13 = pow(1e3, 5.0/6.0) * pow(1e9, 7.0/6.0) */
		constexpr const value_type c = weif::math::Kolmogorov_Cn2_scale<value_type> * (16 * 1e13) * PI * PI;

		const value_type fresnel_radius = sqrt(lambda() * altitude);
		const value_type z = (static_cast<value_type>(1) / (static_cast<value_type>(1) + aperture_scale() / fresnel_radius) - grid_.origin()) / grid_.delta();

		return c * pow(altitude, static_cast<value_type>(5.0/6.0)) / pow(lambda(), static_cast<value_type>(7.0/6.0)) * wf_(z);
	}

public:
	template<class E>
	weight_function_base(value_type lambda, value_type aperture_scale, const uniform_grid<value_type>& grid, const xt::xexpression<E>& values):
		lambda_{lambda},
		aperture_scale_{aperture_scale},
		grid_{grid},
		wf_{values, first_order_boundary<value_type>{0, 0}} {}

	const auto& lambda() const noexcept { return lambda_; /* nm */ }
	const auto& aperture_scale() const noexcept { return aperture_scale_; /* mm */ }
};

template<class SF, class AF, class E>
auto dimensionless_weight_function(SF&& spectral_filter, AF&& aperture_filter, E&& e) noexcept {
	using namespace std::placeholders;
	using boost::math::quadrature::exp_sinh;
	using value_type = xt::get_value_type_t<std::decay_t<E>>;

	auto spectrum_fcnt = [
		spectral_filter = std::forward<SF>(spectral_filter),
		aperture_filter = std::forward<AF>(aperture_filter)
	] (value_type u, value_type x) noexcept -> value_type {
		using namespace std;

		const auto t = pow(u, static_cast<value_type>(8.0/3.0));

		if (t == static_cast<value_type>(0)) {
			return static_cast<value_type>(0);
		}

		return spectral_filter(u * u) * aperture_filter(x * u) / t;
	};

	/* exp-sinh quadrature works poorly for higher altutudes due to
	 * $\sin^2(\pi u^2)$ term. Tanaka, et al. (doi: 10.1007/s00211-008-0195-1)
	 * reveal the reason through the complex plane where $\sin^2(\pi z^2)$
	 * is unbounded in $D_{DE,3}$. However, it seems that there are
	 * alternative DE and SE quadratures which could work better.
	 */
	auto integrator = std::make_unique<exp_sinh<value_type>>();

	auto fcnt = [
		integrator = std::move(integrator),
		spectrum_fcnt = std::move(spectrum_fcnt)
	] (value_type z) -> value_type {
		const auto tol = std::pow(std::numeric_limits<value_type>::epsilon(), static_cast<value_type>(2.0/3.0));
		const auto x = (static_cast<value_type>(1) - z) / z;

		return integrator->integrate(std::bind(std::cref(spectrum_fcnt), _1, x), tol);
	};

	return xt::make_lambda_xfunction(std::move(fcnt), std::forward<E>(e));
}

template<class SF, class AF, class E>
auto dimensionless_weight_function_2d(SF&& spectral_filter, AF&& aperture_filter, E&& e) noexcept {
	using namespace std::placeholders;
	using boost::math::quadrature::exp_sinh;
	using boost::math::quadrature::tanh_sinh;
	using value_type = xt::get_value_type_t<std::decay_t<E>>;

	auto axial_integrator = std::make_unique<tanh_sinh<value_type>>();

	auto spectrum_fcnt_axial = [
		aperture_filter = std::forward<AF>(aperture_filter)
	] (value_type u, value_type x, value_type phi, value_type theta) noexcept -> value_type {
		using namespace std;
		using namespace boost::math;

		const auto xu = x * u;

		if (isinf(xu)) {
			return aperture_filter(xu, static_cast<value_type>(0));
		}

		const auto c = abs(phi) < static_cast<value_type>(0.5) ? cos_pi(phi) : -cos_pi(theta);
		const auto s = abs(phi) < static_cast<value_type>(0.5) ? sin_pi(phi) : sin_pi(theta);

		return aperture_filter(xu * c, xu * s);
	};

	auto spectrum_fcnt = [
		axial_integrator = std::move(axial_integrator),
		spectral_filter = std::forward<SF>(spectral_filter),
		spectrum_fcnt_axial = std::move(spectrum_fcnt_axial)
	] (value_type u, value_type x) noexcept -> value_type {
		using namespace std;

		const auto t = pow(u, static_cast<value_type>(8.0/3.0));

		if (t == static_cast<value_type>(0)) {
			return static_cast<value_type>(0);
		}

		const auto tol = std::pow(std::numeric_limits<value_type>::epsilon(), static_cast<value_type>(2.0/3.0));
		const auto af = axial_integrator->integrate(std::bind(std::cref(spectrum_fcnt_axial), u, x, _1, _2), tol);

		return spectral_filter(u * u) * af / t;
	};

	auto radial_integrator = std::make_unique<exp_sinh<value_type>>();

	auto fcnt = [
		radial_integrator = std::move(radial_integrator),
		spectrum_fcnt = std::move(spectrum_fcnt)
	] (value_type z) -> value_type {
		const auto tol = std::pow(std::numeric_limits<value_type>::epsilon(), static_cast<value_type>(2.0/3.0));
		const auto x = (static_cast<value_type>(1) - z) / z;

		return radial_integrator->integrate(std::bind(std::cref(spectrum_fcnt), _1, x), tol);
	};

	return xt::make_lambda_xfunction(std::move(fcnt), std::forward<E>(e)) * static_cast<value_type>(0.5);
}

} // detail
} // weif

#endif // _WEIF_DETAIL_WEIGHT_FUNCTION_BASE_H

/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_ANGLE_AVERAGED_H
#define _WEIF_AF_ANGLE_AVERAGED_H

#include <cstdlib>
#include <utility>

#include <boost/math/quadrature/tanh_sinh.hpp>

#include <weif/detail/cubic_spline.h>
#include <weif/uniform_grid.h>
#include <weif_export.h>


namespace weif {
namespace af {

template<class T>
class WEIF_EXPORT angle_averaged {
public:
	using value_type = T;

private:
	uniform_grid<value_type> grid_;
	detail::cubic_spline<value_type> af_;

	template<class AF>
	static auto integrate_aperture_function(AF&& aperture_filter, std::size_t size) {
		using boost::math::quadrature::tanh_sinh;

		auto integrator = std::make_unique<tanh_sinh<value_type>>();

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return xt::make_lambda_xfunction([integrator = std::move(integrator), aperture_filter = std::forward<AF>(aperture_filter)] (value_type z) -> value_type {
			if (z == static_cast<value_type>(0))
				return static_cast<value_type>(0);

			const auto tol = std::pow(std::numeric_limits<value_type>::epsilon(), static_cast<value_type>(2.0/3.0));
			const auto u = (static_cast<value_type>(1) - z) / z;

			return integrator->integrate([u, &aperture_filter] (value_type t) noexcept {
				using namespace std;

				const auto f = PI * (t + static_cast<value_type>(1));
				const auto ux = u * cos(f);
				const auto uy = u * sin(f);

				return aperture_filter(ux, uy);
			}, tol) / 2;
		}, xt::linspace(static_cast<value_type>(0), static_cast<value_type>(1), size));
	}

	template<class E>
	explicit angle_averaged(const xt::xexpression<E>& values):
		grid_{static_cast<value_type>(0), static_cast<value_type>(1) / (values.derived_cast().size() - 1), values.derived_cast().size()},
		af_{values, typename detail::cubic_spline<T>::first_order_boundary{0, 0}} {}

public:
	template<class AF>
	angle_averaged(AF&& aperture_filter, std::size_t size):
		angle_averaged(integrate_aperture_function(std::forward<AF>(aperture_filter), size)) {}

	value_type operator() (value_type u) const noexcept {
		const value_type z = (static_cast<value_type>(1) / (static_cast<value_type>(1) + u) - grid_.origin()) / grid_.delta();

		return af_(z);
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) { return this->operator()(x); }, e.derived_cast());
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E2>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

} // af
} // weif

#endif // _WEIF_AF_ANGLE_AVERAGED_H

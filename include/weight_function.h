#ifndef _WEIF_WEIGHT_FUNCTION_H
#define _WEIF_WEIGHT_FUNCTION_H

#include <cmath>
#include <functional>
#include <limits>
#include <utility>

#include <xtensor/xmath.hpp>

#include <cubic_spline.h>
#include <uniform_grid.h>

#include <weif_export.h>


namespace weif {

template<class T>
class WEIF_EXPORT weight_function {
public:
	static_assert(std::is_floating_point<T>::value, "type T is not supported");

	using value_type = T;

private:
	using function_type = std::function<value_type(value_type, value_type)>;

private:
	value_type lambda_;
	value_type aperture_scale_;
	uniform_grid<value_type> grid_;
	cubic_spline<value_type> wf_;

private:
	static auto make_int(function_type&& fun, std::size_t size);

	template<class E>
	weight_function(value_type lambda, value_type aperture_scale, const xt::xexpression<E>& values);
	weight_function(value_type lambda, value_type aperture_scale, function_type&& fun, std::size_t size);

public:
	template<class SF, class AF>
	weight_function(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, std::size_t size):
		weight_function(lambda, aperture_scale,
			[spectral_filter = std::forward<SF>(spectral_filter), aperture_filter = std::forward<AF>(aperture_filter)] (value_type u, value_type x) noexcept -> value_type {
			if (u == static_cast<value_type>(0) || u == std::numeric_limits<value_type>::infinity())
				return static_cast<value_type>(0);

			if (u < static_cast<value_type>(1)) {
				return std::pow(u, static_cast<value_type>(4.0/3.0)) * spectral_filter.regular(u * u) * aperture_filter(x * u);
			}

			return std::pow(u, -static_cast<value_type>(8.0/3.0)) * spectral_filter(u * u) * aperture_filter(x * u);
		}, size) {}

	const auto& lambda() const noexcept { return lambda_; /* nm */ }
	const auto& aperture_scale() const noexcept { return aperture_scale_; /* mm */ }

	inline value_type operator() (value_type altitude) const noexcept {
		using namespace std;

		constexpr const auto PI = xt::numeric_constants<value_type>::PI;
		/* 1e13 = pow(1e3, 5.0/6.0) * pow(1e9, 7.0/6.0) */
		constexpr const value_type c = 9.69e-3 * 32 * PI * PI * PI * 1e13;

		if (altitude == static_cast<value_type>(0))
			return static_cast<value_type>(0);

		const value_type fresnel_radius = sqrt(lambda() * altitude);
		const value_type z = (fresnel_radius / (fresnel_radius + aperture_scale()) - grid_.origin()) / grid_.delta();

		return c / pow(lambda(), static_cast<value_type>(7.0/6.0)) * pow(altitude, static_cast<value_type>(5.0/6.0)) * wf_(z);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->operator()(x);
		}, e.derived_cast());
	}
};

} // weif

#endif // _WEIF_WEIGHT_FUNCTION_H

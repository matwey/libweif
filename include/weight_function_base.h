#ifndef _WEIF_WEIGHT_FUNCTION_BASE_H
#define _WEIF_WEIGHT_FUNCTION_BASE_H

#include <cmath>
#include <functional>

#include <xtensor/xexpression.hpp>
#include <xtensor/xmath.hpp>

#include <cubic_spline.h>
#include <uniform_grid.h>

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
		constexpr const value_type c = 9.69e-3 * 16 * PI * PI * 1e13;

		if (altitude == static_cast<value_type>(0))
			return static_cast<value_type>(0);

		const value_type fresnel_radius = sqrt(lambda() * altitude);
		const value_type z = (fresnel_radius / (fresnel_radius + aperture_scale()) - grid_.origin()) / grid_.delta();

		return c * pow(altitude, static_cast<value_type>(5.0/6.0)) / pow(lambda(), static_cast<value_type>(7.0/6.0)) * wf_(z);
	}

public:
	template<class E>
	weight_function_base(value_type lambda, value_type aperture_scale, const xt::xexpression<E>& values):
		lambda_{lambda},
		aperture_scale_{aperture_scale},
		grid_{static_cast<value_type>(0),
			static_cast<value_type>(1) / (values.derived_cast().size()-1),
			values.derived_cast().size()},
		wf_{values, typename cubic_spline<T>::first_order_boundary{0, 0}} {}

	const auto& lambda() const noexcept { return lambda_; /* nm */ }
	const auto& aperture_scale() const noexcept { return aperture_scale_; /* mm */ }
};

} // detail
} // weif

#endif // _WEIF_WEIGHT_FUNCTION_BASE_H

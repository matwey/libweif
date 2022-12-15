#ifndef _WEIF_WEIGHT_FUNCTION_2D_H
#define _WEIF_WEIGHT_FUNCTION_2D_H

#include <type_traits>

#include <xtensor/xexpression.hpp>
#include <xtensor/xmath.hpp>

#include <weight_function_base.h>

#include <weif_export.h>


namespace weif {

template<class T>
class WEIF_EXPORT weight_function_2d:
	public detail::weight_function_base<T> {
public:
	using value_type = typename detail::weight_function_base<T>::value_type;

private:
	using function_type = std::function<value_type(value_type, value_type, value_type)>;

private:
	static auto integrate_weight_function(function_type&& fun, std::size_t size);

	weight_function_2d(value_type lambda, value_type aperture_scale, function_type&& fun, std::size_t size);

public:
	template<class SF, class AF>
	weight_function_2d(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, std::size_t size):
		weight_function_2d(lambda, aperture_scale,
			[spectral_filter = std::forward<SF>(spectral_filter), aperture_filter = std::forward<AF>(aperture_filter)] (value_type ux, value_type uy, value_type x) noexcept -> value_type {
			using namespace std;

			if (ux == static_cast<value_type>(0) && uy == static_cast<value_type>(0))
				return static_cast<value_type>(0);

			if (ux == std::numeric_limits<value_type>::infinity() || uy == std::numeric_limits<value_type>::infinity())
				return static_cast<value_type>(0);

			const auto u2 = ux * ux + uy * uy;

			if (u2 < static_cast<value_type>(1)) {
				return pow(u2, static_cast<value_type>(1.0/6.0)) * spectral_filter.regular(u2) * aperture_filter(x * ux, x * uy);
			}

			const auto t = pow(u2, -static_cast<value_type>(11.0/6.0));
			if (t == static_cast<value_type>(0))
				return static_cast<value_type>(0);

			return t * spectral_filter(u2) * aperture_filter(x * ux, x * uy);
		}, size) {}

	inline value_type operator() (value_type altitude) const noexcept {
		return detail::weight_function_base<T>::operator()(altitude);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->operator()(x);
		}, e.derived_cast());
	}
};

} // weif

#endif // _WEIF_WEIGHT_FUNCTION_2D_H

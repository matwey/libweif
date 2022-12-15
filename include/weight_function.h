#ifndef _WEIF_WEIGHT_FUNCTION_H
#define _WEIF_WEIGHT_FUNCTION_H

#include <type_traits>

#include <xtensor/xexpression.hpp>
#include <xtensor/xmath.hpp>

#include <weight_function_base.h>

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

	weight_function(value_type lambda, value_type aperture_scale, function_type&& fun, std::size_t size);

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

} // weif

#endif // _WEIF_WEIGHT_FUNCTION_H

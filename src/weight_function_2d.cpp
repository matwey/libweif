#include <cmath>
#include <limits>

#include <boost/math/quadrature/exp_sinh.hpp>

#include <weight_function_2d.h>


namespace weif {

template<class T>
auto weight_function_2d<T>::integrate_weight_function(function_type&& fun, std::size_t size) {
	auto integrator = std::make_unique<boost::math::quadrature::exp_sinh<value_type>>();

	return xt::make_lambda_xfunction([integrator = std::move(integrator), fun = std::forward<function_type>(fun)] (value_type z) -> value_type {
		using namespace std::placeholders;

		if (z == static_cast<value_type>(0))
			return static_cast<value_type>(0);

		const auto tol = std::pow(std::numeric_limits<value_type>::epsilon(), static_cast<value_type>(2.0/3.0));
		const auto x = (static_cast<value_type>(1) - z) / z;

		return integrator->integrate([&integrator, &fun, &x, &tol] (value_type ux) noexcept {
			return integrator->integrate(std::bind(std::cref(fun), ux, _1, x), tol);
		}, tol) * 4;
	}, xt::linspace(static_cast<value_type>(0), static_cast<value_type>(1), size));
}

template<class T>
weight_function_2d<T>::weight_function_2d(value_type lambda, value_type aperture_scale, function_type&& fun, std::size_t size):
	detail::weight_function_base<T>(lambda, aperture_scale, integrate_weight_function(std::forward<function_type>(fun), size)) {}


template class weight_function_2d<float>;
template class weight_function_2d<double>;
template class weight_function_2d<long double>;

} // weif

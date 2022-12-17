#include <cmath>
#include <limits>

#include <boost/math/quadrature/exp_sinh.hpp>

#include <weight_function.h>


namespace weif {

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

template<class T>
weight_function<T>::weight_function(value_type lambda, value_type aperture_scale, function_type&& fun, std::size_t size):
	detail::weight_function_base<T>(lambda, aperture_scale, integrate_weight_function(std::forward<function_type>(fun), size)) {}


template class weight_function<float>;
template class weight_function<double>;
template class weight_function<long double>;

} // weif

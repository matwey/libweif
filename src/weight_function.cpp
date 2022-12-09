#include <limits>

#include <boost/math/quadrature/exp_sinh.hpp>

#include <xtensor/xio.hpp>

#include <error.h>
#include <weight_function.h>


namespace weif {

template<class T>
auto weight_function<T>::make_int(function_type&& fun, std::size_t size) {
	auto integrator = std::make_unique<boost::math::quadrature::exp_sinh<value_type>>();

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
template<class E>
weight_function<T>::weight_function(typename weight_function<T>::value_type lambda, typename weight_function<T>::value_type aperture_scale, const xt::xexpression<E>& values):
	lambda_{lambda},
	aperture_scale_{aperture_scale},
	grid_{static_cast<value_type>(0), static_cast<value_type>(1) / (values.derived_cast().size()-1), values.derived_cast().size()},
	wf_{values, typename cubic_spline<T>::first_order_boundary{0, 0}} {}

template<class T>
weight_function<T>::weight_function(typename weight_function<T>::value_type lambda, typename weight_function<T>::value_type aperture_scale, function_type&& fun, std::size_t size):
	weight_function(lambda, aperture_scale, make_int(std::forward<function_type>(fun), size)) {}


template class weight_function<float>;
template class weight_function<double>;
template class weight_function<long double>;

} // weif


#include <limits>

#include <gsl/gsl_integration.h>

#include <xtensor/xio.hpp>

#include <error.h>
#include <weight_function.h>


namespace weif {

template<class T>
auto weight_function<T>::make_int(function_type&& fun, std::size_t size) {
	constexpr std::size_t intervals = 1024;

	auto workspace_ptr = gsl_integration_workspace_alloc(intervals);
	if (workspace_ptr == nullptr) {
		throw std::bad_alloc();
	}

	std::unique_ptr<gsl_integration_workspace, void (*)(gsl_integration_workspace*)> workspace{workspace_ptr, &gsl_integration_workspace_free};

	return xt::make_lambda_xfunction([workspace = std::move(workspace), fun = std::move(fun)] (const auto& z) {
		using namespace std::placeholders;

		if (z == static_cast<value_type>(0))
			return static_cast<value_type>(0);

		const auto x = (static_cast<value_type>(1) - z) / z;
		auto binder = std::bind(std::cref(fun), _1, static_cast<double>(x));

		gsl_function f;
		f.function = [] (double u, void* params) -> double {
			const auto fn = reinterpret_cast<decltype(binder)*>(params);

			return (*fn)(u);
		};
		f.params = &binder;

		double result;
		double abserr;

		int status = gsl_integration_qagiu(&f, 0.0, std::numeric_limits<value_type>::epsilon(), std::numeric_limits<value_type>::epsilon(), intervals, workspace.get(), &result, &abserr);
		if (status) {
			throw gsl_error(status);
		}

		return static_cast<value_type>(result);
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

} // weif


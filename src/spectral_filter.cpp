#include <array>
#include <cassert>
#include <complex>
#include <memory>
#include <limits>

#include <fftw3.h>

#include <gsl/gsl_integration.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <spectral_filter.h>


namespace weif {

template<class T>
template<class E>
xt::xtensor<std::complex<typename spectral_filter<T>::value_type>, 1> spectral_filter<T>::make_fft(const xt::xexpression<E>& e) {
	const auto size = e.derived_cast().size();

	xt::xtensor<std::complex<value_type>, 1> ret{std::array{size / 2 + 1}};
	auto in_adapter = xt::adapt(reinterpret_cast<value_type*>(ret.data()), size, xt::no_ownership(), std::array{size});
	in_adapter.assign(e);

	if constexpr (std::is_same<value_type, float>::value) {
		fftwf_plan p = fftwf_plan_dft_r2c_1d(size,
			in_adapter.data(),
			reinterpret_cast<fftwf_complex*>(ret.data()),
			FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

		assert(p != nullptr);

		fftwf_execute(p);
		fftwf_destroy_plan(p);
	} else if (std::is_same<value_type, double>::value) {
		fftw_plan p = fftw_plan_dft_r2c_1d(size,
			in_adapter.data(),
			reinterpret_cast<fftw_complex*>(ret.data()),
			FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

		assert(p != nullptr);

		fftw_execute(p);
		fftw_destroy_plan(p);
	} else {
		static_assert(true, "value_type is not supported");
	}

	/* Boundary condition at +inf */
	ret(ret.size()-1) = std::complex<value_type>{0, 0};

	return ret;
}

template<class T>
template<class E1, class E2>
spectral_filter<T>::spectral_filter(const uniform_grid<value_type>& grid, const xt::xexpression<E1>& real, const xt::xexpression<E2>& imag, value_type carrier):
	grid_{grid},
	real_{real.derived_cast(), typename cubic_spline<T>::first_order_boundary{0, 0}},
	imag_{imag.derived_cast(), typename cubic_spline<T>::second_order_boundary{0, 0}},
	carrier_{carrier},
	equiv_lambda_{eval_equiv_lambda()} {}

template<class T>
template<class E>
spectral_filter<T>::spectral_filter(value_type delta, const xt::xexpression<E>& e, T carrier):
	spectral_filter(uniform_grid{static_cast<value_type>(0), delta, e.derived_cast().size()},
		xt::real(e.derived_cast()),
		xt::imag(e.derived_cast()),
		carrier) {}

template<class T>
spectral_filter<T>::spectral_filter(const spectral_response<value_type>& response, std::size_t size, std::size_t carrier_idx, std::size_t padded_size):
	spectral_filter(static_cast<value_type>(1) / response.grid().delta() / padded_size,
		make_fft(xt::view(
			xt::tile(xt::pad(response.data() / response.grid().values(),
				std::vector<std::size_t>{{0, (padded_size > response.grid().size() ? padded_size - response.grid().size() : 0)}}), {2}),
			xt::range(carrier_idx, carrier_idx + padded_size))),
		response.grid().values()[carrier_idx]) {}

template<class T>
spectral_filter<T>::spectral_filter(const spectral_response<value_type>& response, std::size_t size, value_type carrier):
	spectral_filter(response, size, response.grid().to_index(carrier), std::max(response.grid().size(), size)) {}

template<class T>
spectral_filter<T>::spectral_filter(const spectral_response<value_type>& response, std::size_t size):
	spectral_filter(response, size, response.effective_lambda()) {}

template<class T>
void spectral_filter<T>::normalize() noexcept {
	const auto lambda_0 = equiv_lambda();

	grid_ *= lambda_0;
	carrier_ /= lambda_0;
	equiv_lambda_ /= lambda_0;
	real_ *= lambda_0;
	imag_ *= lambda_0;
}

template<class T>
spectral_filter<T> spectral_filter<T>::normalized() const {
	spectral_filter<T> ret{*this};

	ret.normalize();

	return ret;
}

template<class T>
typename spectral_filter<T>::value_type spectral_filter<T>::eval_equiv_lambda() const {
	constexpr std::size_t intervals = 1024;

	auto workspace_ptr = gsl_integration_workspace_alloc(intervals);
	if (workspace_ptr == nullptr) {
		throw std::bad_alloc();
	}

	std::unique_ptr<gsl_integration_workspace, void (*)(gsl_integration_workspace*)> workspace{workspace_ptr, &gsl_integration_workspace_free};

	gsl_function f;
	f.function = [] (double x, void* params) -> double {
		const auto that = reinterpret_cast<const spectral_filter<T>*>(params);

		if (x == 0.0)
			return 0.0;

		return std::pow(x, -11.0/6.0) * that->operator()(x);
	};
	f.params = const_cast<void*>(reinterpret_cast<const void*>(this));

	double result;
	double abserr;

	int status = gsl_integration_qagiu(&f, 0.0, std::numeric_limits<value_type>::epsilon(), std::numeric_limits<value_type>::epsilon(), intervals, workspace.get(), &result, &abserr);
	if (status) {
		throw gsl_error(status);
	}

	return static_cast<value_type>(3.28) * std::pow(static_cast<value_type>(result), -static_cast<value_type>(6.0/7.0));
}

template class spectral_filter<float>;
template class spectral_filter<double>;

} // weif

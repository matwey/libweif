#include <array>
#include <cassert>
#include <complex>
#include <memory>
#include <limits>

#include <boost/math/quadrature/exp_sinh.hpp>

#include <fftw3.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <fftw3_wrap.h>
#include <spectral_filter.h>


namespace weif {

template<class T>
template<class E>
xt::xtensor<std::complex<typename spectral_filter<T>::value_type>, 1> spectral_filter<T>::make_fft(const xt::xexpression<E>& e) {
	const auto size = e.derived_cast().size();

	xt::xtensor<std::complex<value_type>, 1> ret{std::array{size / 2 + 1}};
	auto in_adapter = xt::adapt(reinterpret_cast<value_type*>(ret.data()), size, xt::no_ownership(), std::array{size});
	in_adapter.assign(e);

	fft_plan_r2c plan{std::array{static_cast<int>(size)}, in_adapter.data(), ret.data(), FFTW_ESTIMATE | FFTW_DESTROY_INPUT};
	plan(in_adapter.data(), ret.data());

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
	boost::math::quadrature::exp_sinh<value_type> integrator;

	const auto i = integrator.integrate([this] (value_type x) {
		if (x == static_cast<value_type>(0.0) || x == std::numeric_limits<value_type>::infinity())
			return static_cast<value_type>(0.0);

		if (x < static_cast<value_type>(1)) {
			return std::pow(x, static_cast<value_type>(1.0/6.0)) * this->regular(x);
		}

		return std::pow(x, -static_cast<value_type>(11.0/6.0)) * this->operator()(x);
	});

	return static_cast<value_type>(3.28) * std::pow(i, -static_cast<value_type>(6.0/7.0));
}

template class spectral_filter<float>;
template class spectral_filter<double>;
template class spectral_filter<long double>;

} // weif

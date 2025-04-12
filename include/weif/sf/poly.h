/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_SF_POLY_H
#define _WEIF_SF_POLY_H

#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>

#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/special_functions/sinc.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <weif/detail/cubic_spline.h>
#include <weif/detail/fftw3_wrap.h> // IWYU pragma: keep
#include <weif/uniform_grid.h>
#include <weif/spectral_response.h>
#include <weif_export.h>


namespace weif {
namespace sf {

template<class T>
class WEIF_EXPORT poly {
public:
	using value_type = T;

private:
	/* Consider using different type for uniform_grid, for instance, boost::cpp_dec_float */
	uniform_grid<value_type> grid_;
	detail::cubic_spline<value_type> real_;
	detail::cubic_spline<value_type> imag_;
	value_type carrier_;
	value_type equiv_lambda_;

	value_type eval_equiv_lambda() const;

	template<class E>
	xt::xtensor<std::complex<value_type>, 1> make_fft(const xt::xexpression<E>& e);

	template<class E1, class E2>
	poly(const uniform_grid<value_type>& grid, const xt::xexpression<E1>& real, const xt::xexpression<E2>& imag, value_type carrier):
		grid_{grid},
		real_{real.derived_cast(), typename detail::first_order_boundary<value_type>{0, 0}},
		imag_{imag.derived_cast(), typename detail::second_order_boundary<value_type>{0, 0}},
		carrier_{carrier},
		equiv_lambda_{eval_equiv_lambda()} {}

	template<class E>
	poly(value_type delta, const xt::xexpression<E>& e, value_type carrier):
		poly({static_cast<value_type>(0), delta, e.derived_cast().size()},
			xt::real(e.derived_cast()),
			xt::imag(e.derived_cast()),
			carrier) {}

	poly(const spectral_response<value_type>& response, std::size_t carrier_idx, std::size_t padded_size):
		poly(static_cast<value_type>(1) / response.grid().delta() / padded_size,
			make_fft(xt::view(
				xt::tile(xt::pad(response.data() / response.grid().values(),
					std::vector<std::size_t>{{0, (padded_size > response.grid().size() ? padded_size - response.grid().size() : 0)}}), {2}),
				xt::range(carrier_idx, carrier_idx + padded_size))),
			response.grid().values()[carrier_idx]) {}

public:
	poly(const spectral_response<value_type>& response, std::size_t size, value_type carrier):
		poly(response, response.grid().to_index(carrier), std::max(response.grid().size(), size)) {}
	poly(const spectral_response<value_type>& response, std::size_t size):
		poly(response, size, response.effective_lambda()) {}

	value_type operator() (const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;

		const auto ax = abs(x);

		if (grid() <= ax)
			return static_cast<value_type>(0);

		constexpr auto PI = xt::numeric_constants<value_type>::PI;
		const auto c = PI * carrier();
		const auto cx = ax * c;
		const auto dx = (ax / static_cast<value_type>(2) - grid().origin()) / grid().delta();

		/*
		 * Implementation note
		 *
		 * Tokovinin (2003), and Kornilov and Safonov (2019) noted that
		 * the spectral filter is a squared imaginary and real parts of
		 * Fourier transform for the spectral response. Because of
		 * squaring, it does not matter which sign they used in the
		 * definition of the Fourier transform.
		 *
		 * However, when the Fourier shift theorem is applied the sign
		 * must be consistent with FFTW3 real-to-complex routines used
		 * to perform actual Fourier transform. The FFTW3 uses a minus
		 * sign for the forward Fourier transform. Note, that atmos
		 * software uses the opposite sign.
		 */

		return pow(sin(cx) * real()(dx) - cos(cx) * imag()(dx), 2);
	}

	value_type regular(const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;
		using boost::math::sinc_pi;

		const auto ax = abs(x);

		if (grid() <= ax)
			return static_cast<value_type>(0);

		constexpr auto PI = xt::numeric_constants<value_type>::PI;
		const auto c = PI * carrier();
		const auto cx = ax * c;
		const auto dx = (ax / static_cast<value_type>(2) - grid().origin()) / grid().delta();
		const auto im = (dx < static_cast<value_type>(1) ?
			(imag().values()(1) + imag().double_primes()(1) * (dx * dx - static_cast<value_type>(1)) / static_cast<value_type>(6)) / (grid().delta() * static_cast<value_type>(2)) :
			imag()(dx) / ax);

		return pow(c * sinc_pi(cx) * real()(dx) - cos(cx) * im, 2);
	}

	template<class E, xt::enable_xexpression<std::decay_t<E>, bool> = true>
	auto operator() (E&& e) const noexcept {
		return xt::make_lambda_xfunction([this] (auto x) -> decltype(x) {
			return this->operator()(x);
		}, std::forward<E>(e));
	}

	template<class E, xt::enable_xexpression<std::decay_t<E>, bool> = true>
	auto regular(E&& e) const noexcept {
		return xt::make_lambda_xfunction([this] (auto x) -> decltype(x) {
			return this->regular(x);
		}, std::forward<E>(e));
	}

	const auto& grid() const noexcept { return grid_; }
	const auto& real() const noexcept { return real_; }
	const auto& imag() const noexcept { return imag_; }
	const auto& carrier() const noexcept { return carrier_; }
	const auto& equiv_lambda() const noexcept { return equiv_lambda_; }

	poly<value_type>& normalize() noexcept;
	poly<value_type> normalized() const;
};

template<class T>
template<class E>
xt::xtensor<std::complex<typename poly<T>::value_type>, 1> poly<T>::make_fft(const xt::xexpression<E>& e) {
	const auto size = e.derived_cast().size();

	xt::xtensor<std::complex<value_type>, 1> ret{std::array{size / 2 + 1}};
	auto in_adapter = xt::adapt(reinterpret_cast<value_type*>(ret.data()), size, xt::no_ownership(), std::array{size});
	in_adapter.assign(e);

	detail::fft_plan_r2c plan{std::array{static_cast<int>(size)}, in_adapter.data(), ret.data(), FFTW_ESTIMATE | FFTW_DESTROY_INPUT};
	plan(in_adapter.data(), ret.data());

	/* Boundary condition at +inf */
	ret(ret.size()-1) = std::complex<value_type>{0, 0};

	return ret;
}

template<class T>
poly<T>& poly<T>::normalize() noexcept {
	const auto lambda_0 = equiv_lambda();

	grid_ *= lambda_0;
	carrier_ /= lambda_0;
	equiv_lambda_ /= lambda_0;
	real_ *= lambda_0;
	imag_ *= lambda_0;

	return *this;
}

template<class T>
poly<T> poly<T>::normalized() const {
	poly<T> ret{*this};

	ret.normalize();

	return ret;
}

template<class T>
typename poly<T>::value_type poly<T>::eval_equiv_lambda() const {
	using namespace std;

	boost::math::quadrature::exp_sinh<value_type> integrator;

	const auto i = integrator.integrate([this] (value_type x) {
		if (x == static_cast<value_type>(0.0) || x == std::numeric_limits<value_type>::infinity())
			return static_cast<value_type>(0.0);

		if (x < static_cast<value_type>(1)) {
			return pow(x, static_cast<value_type>(1.0/6.0)) * this->regular(x);
		}

		return pow(x, -static_cast<value_type>(11.0/6.0)) * this->operator()(x);
	});

	return static_cast<value_type>(3.28) * pow(i, -static_cast<value_type>(6.0/7.0));
}


template<class T>
poly(const spectral_response<T>& response, std::size_t size, T carrier) -> poly<T>;
template<class T>
poly(const spectral_response<T>& response, std::size_t size) -> poly<T>;


extern template class poly<float>;
extern template class poly<double>;
extern template class poly<long double>;

} // sf
} // weif

#endif // _WEIF_SF_POLY_H



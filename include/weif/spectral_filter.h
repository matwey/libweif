/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_SPECTRAL_FILTER_H
#define _WEIF_SPECTRAL_FILTER_H

#include <algorithm>
#include <array> // IWYU pragma: keep
#include <cmath>
#include <complex>
#include <limits>
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

template<class T>
class WEIF_EXPORT mono_spectral_filter {
public:
	using value_type = T;

	value_type operator() (const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;

		const auto ax = abs(x);

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(sin(PI * ax), 2);
	}

	value_type regular(const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;
		using boost::math::sinc_pi;

		const auto ax = abs(x);

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(PI * sinc_pi(PI * ax), 2);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->operator()(x);
		}, e.derived_cast());
	}

	template<class E>
	auto regular(const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->regular(x);
		}, e.derived_cast());
	}
};

template<class T>
class WEIF_EXPORT spectral_filter {
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
	spectral_filter(const uniform_grid<value_type>& grid, const xt::xexpression<E1>& real, const xt::xexpression<E2>& imag, value_type carrier):
		grid_{grid},
		real_{real.derived_cast(), typename detail::cubic_spline<T>::first_order_boundary{0, 0}},
		imag_{imag.derived_cast(), typename detail::cubic_spline<T>::second_order_boundary{0, 0}},
		carrier_{carrier},
		equiv_lambda_{eval_equiv_lambda()} {}

	template<class E>
	spectral_filter(value_type delta, const xt::xexpression<E>& e, value_type carrier):
		spectral_filter({static_cast<value_type>(0), delta, e.derived_cast().size()},
			xt::real(e.derived_cast()),
			xt::imag(e.derived_cast()),
			carrier) {}

	spectral_filter(const spectral_response<value_type>& response, std::size_t size, std::size_t carrier_idx, std::size_t padded_size):
		spectral_filter(static_cast<value_type>(1) / response.grid().delta() / padded_size,
			make_fft(xt::view(
				xt::tile(xt::pad(response.data() / response.grid().values(),
					std::vector<std::size_t>{{0, (padded_size > response.grid().size() ? padded_size - response.grid().size() : 0)}}), {2}),
				xt::range(carrier_idx, carrier_idx + padded_size))),
			response.grid().values()[carrier_idx]) {}

public:
	spectral_filter(const spectral_response<value_type>& response, std::size_t size, value_type carrier):
		spectral_filter(response, size, response.grid().to_index(carrier), std::max(response.grid().size(), size)) {}
	spectral_filter(const spectral_response<value_type>& response, std::size_t size):
		spectral_filter(response, size, response.effective_lambda()) {}

	const auto& grid() const noexcept { return grid_; }
	const auto& real() const noexcept { return real_; }
	const auto& imag() const noexcept { return imag_; }
	const auto& carrier() const noexcept { return carrier_; }
	const auto& equiv_lambda() const noexcept { return equiv_lambda_; }

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

		return pow(cos(cx) * imag()(dx) + sin(cx) * real()(dx), 2);
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

		return pow(cos(cx) * im + c * sinc_pi(cx) * real()(dx), 2);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->operator()(x);
		}, e.derived_cast());
	}

	template<class E>
	auto regular(const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->regular(x);
		}, e.derived_cast());
	}

	void normalize() noexcept;
	spectral_filter<value_type> normalized() const;
};

template<class T>
template<class E>
xt::xtensor<std::complex<typename spectral_filter<T>::value_type>, 1> spectral_filter<T>::make_fft(const xt::xexpression<E>& e) {
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


template<class T>
spectral_filter(const spectral_response<T>& response) -> spectral_filter<T>;
template<class T>
spectral_filter(const uniform_grid<T>& grid, const xt::xtensor<T, 1>& data) -> spectral_filter<T>;


extern template class spectral_filter<float>;
extern template class spectral_filter<double>;
extern template class spectral_filter<long double>;

} // weif

#endif // _WEIF_SPECTRAL_FILTER_H

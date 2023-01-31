/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_DIGITAL_FILTER_2D_H
#define _WEIF_DIGITAL_FILTER_2D_H

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp> // IWYU pragma: keep

#include <weif/detail/fftw3_wrap.h>
#include <weif_export.h>

#if __cpp_lib_memory_resource >= 201603
#include <memory_resource>
#endif


namespace weif {

template<class T, class Allocator = std::allocator<T>>
class WEIF_EXPORT digital_filter_2d:
	private Allocator {
public:
	using value_type = T;
	using allocator_type = Allocator;
	using shape_type = std::array<std::size_t, 2>;
	using impulse_type = xt::xtensor<value_type, 2, XTENSOR_DEFAULT_LAYOUT, allocator_type>;

private:
	using function_type = std::function<value_type(value_type, value_type)>;

private:
	impulse_type impulse_;

private:
	static impulse_type make_impulse(function_type&& fun, shape_type shape, const allocator_type& alloc);

	digital_filter_2d(function_type&& fun, shape_type shape, const allocator_type& alloc):
		allocator_type(alloc),
		impulse_{make_impulse(std::forward<function_type>(fun), shape, get_allocator())} {}

public:
	template<class DF>
	digital_filter_2d(DF&& digital_filter_fun, shape_type shape, const allocator_type& alloc = allocator_type()):
		digital_filter_2d(std::function(std::forward<DF>(digital_filter_fun)), shape, alloc) {}

	const allocator_type& get_allocator() const noexcept { return *this; }

	const auto& impulse() const noexcept { return impulse_; }
	const auto& shape()   const noexcept { return impulse_.shape(); }

	value_type operator() (value_type ux, value_type uy) const noexcept {
		using std::cos;
		using std::sin;

		constexpr const auto PI = xt::numeric_constants<value_type>::PI;
		const auto cx = cos(2*PI*ux);
		const auto sx = sin(2*PI*ux);
		const auto cy = cos(2*PI*uy);
		const auto sy = sin(2*PI*uy);

		value_type six = 0;
		value_type cix = 1;
		value_type ret = 0;
		for (std::size_t i = 0; i < std::get<0>(shape()); ++i) {
			const auto i_norm = (i > 0 ? static_cast<value_type>(2) : static_cast<value_type>(1));
			value_type sjy = 0;
			value_type cjy = 1;

			for (std::size_t j = 0; j < std::get<1>(shape()); ++j) {
				const auto j_norm = (j > 0 ? static_cast<value_type>(2) : static_cast<value_type>(1));

				ret += impulse_(i, j) * i_norm * j_norm * (cix * cjy - six * sjy);

				const value_type tmp = cjy * cy - sjy * sy;
				sjy = sjy * cy + cjy * sy;
				cjy = tmp;
			}

			const value_type tmp = cix * cx - six * sx;
			six = six * cx + cix * sx;
			cix = tmp;
		}

		return ret;
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E2>& e2) const noexcept {
		return xt::make_lambda_xfunction([this] (const value_type& ux, const value_type& uy) {
			return this->operator()(ux, uy);
		}, e1.derived_cast(), xt::expand_dims(e2.derived_cast(), 1));
	}
};

template<class T, class Allocator>
typename digital_filter_2d<T, Allocator>::impulse_type
digital_filter_2d<T, Allocator>::make_impulse(function_type&& fun, shape_type shape, const allocator_type& alloc) {
	constexpr value_type nyquist = 0.5;
	const auto& nx = std::get<0>(shape);
	const auto& ny = std::get<1>(shape);
	const auto ux = xt::linspace(static_cast<value_type>(0), nyquist, nx);
	const auto uy = xt::linspace(static_cast<value_type>(0), nyquist, ny);
	const auto fft_norm = static_cast<value_type>(1) /
		static_cast<value_type>(4 * (nx - 1) * (ny - 1));

	impulse_type ret{xt::make_lambda_xfunction(std::forward<function_type>(fun), ux, xt::expand_dims(uy, 1))};

	detail::fft_plan_r2r<T> plan{std::array{static_cast<int>(std::get<0>(shape)), static_cast<int>(std::get<1>(shape))},
		ret.data(), ret.data(), std::array{FFTW_REDFT00, FFTW_REDFT00}, FFTW_ESTIMATE};

	plan(ret.data(), ret.data());

	ret *= fft_norm;

	return ret;
}


extern template class digital_filter_2d<float>;
extern template class digital_filter_2d<double>;
extern template class digital_filter_2d<long double>;

#if __cpp_lib_memory_resource >= 201603

namespace pmr {

template<class T>
using digital_filter_2d = weif::digital_filter_2d<T, std::pmr::polymorphic_allocator<T>>;

} // pmr

extern template class digital_filter_2d<float, std::pmr::polymorphic_allocator<float>>;
extern template class digital_filter_2d<double, std::pmr::polymorphic_allocator<double>>;
extern template class digital_filter_2d<long double, std::pmr::polymorphic_allocator<long double>>;

#endif

} // weif

#endif // _WEIF_DIGITAL_FILTER_2D_H

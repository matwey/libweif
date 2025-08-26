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

#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xtensor.hpp> // IWYU pragma: keep

#include <weif/detail/fftw3_wrap.h>
#include <weif_export.h>

#if __cpp_lib_memory_resource >= 201603
#include <memory_resource>
#endif


namespace weif {

/**
 * @brief Digital filter function
 *
 * @tparam T Numeric type used for calculations
 * @tparam Allocator Memory allocator type (default: std::allocator<T>)
 *
 * Implements a dimensional digital filter.
 */
template<class T, class Allocator = std::allocator<T>>
class WEIF_EXPORT digital_filter_2d:
	private Allocator {
public:
	using value_type = T; ///< Numeric type used for calculations
	using allocator_type = Allocator; ///< Memory allocator type
	using shape_type = std::array<std::size_t, 2>; ///< Filter shape type (Nx, Ny)
	using impulse_type = xt::xtensor<value_type, 2, XTENSOR_DEFAULT_LAYOUT, allocator_type>; ///< Impulse response tensor type

private:
	using function_type = std::function<value_type(value_type, value_type)>;

private:
	impulse_type impulse_;

private:
	static impulse_type make_impulse(function_type&& fun, shape_type shape, const allocator_type& alloc);

	digital_filter_2d(function_type&& fun, shape_type shape, const allocator_type& alloc):
		digital_filter_2d(make_impulse(std::forward<function_type>(fun), shape, get_allocator()), alloc) {}

public:
	/**
	 * @brief Construct from impulse response expression
	 * @tparam E Expression type
	 * @param impulse Input impulse response expression
	 * @param alloc Allocator instance
	 */
	template<class E>
	digital_filter_2d(const xt::xexpression<E>& impulse, const allocator_type& alloc = allocator_type()):
		allocator_type(alloc),
		impulse_{impulse.derived_cast()} {}

	/**
	 * @brief Construct from rvalue impulse response expression
	 * @tparam E Expression type
	 * @param impulse Input impulse response expression (moved)
	 * @param alloc Allocator instance
	 */
	template<class E>
	digital_filter_2d(xt::xexpression<E>&& impulse, const allocator_type& alloc = allocator_type()):
		allocator_type(alloc),
		impulse_{impulse.derived_cast()} {}

	/**
	 * @brief Construct from filter function
	 * @tparam DF Callable filter function type
	 * @param digital_filter_fun Digital filter function
	 * @param shape Filter dimensions (Nx, Ny)
	 * @param alloc Allocator instance
	 *
	 * The digital filter function \f$\Omega(u_x, u_y)\f$ is evaluated on
	 * an appropriate frequency grid and the filter impulse response is
	 * calculated using Fast Fourier Transform. The frequency grid spans
	 * \f$[0, 0.5] \times [0, 0.5]\f$ in dimensionless frequency space.
	 */
	template<class DF>
	digital_filter_2d(DF&& digital_filter_fun, shape_type shape, const allocator_type& alloc = allocator_type()):
		digital_filter_2d(function_type(std::forward<DF>(digital_filter_fun)), shape, alloc) {}

	/// @return Associated allocator
	const allocator_type& get_allocator() const noexcept { return *this; }

	/// @return Reference to impulse response tensor
	const auto& impulse() const noexcept { return impulse_; }

	/// @return Filter dimensions (Nx, Ny)
	const auto& shape()   const noexcept { return impulse_.shape(); }

	/**
	 * @brief Performs in-place amplitude mixing
	 * @return Reference to modified filter
	 *
	 * Modifies the impulse response by subtracting a checkerboard sign
	 * alternation pattern to set the center (0,0) to zero.
	 */
	digital_filter_2d<T, Allocator>& mix() noexcept {
		const auto& nx = std::get<0>(shape());
		const auto& ny = std::get<1>(shape());

		const auto amplitude = impulse_(0,0);

		for (std::size_t i = 0; i < nx; ++i) {
			bool sign = i % 2;
			for (std::size_t j = 0; j < ny; ++j, sign = !sign) {
				impulse_(i, j) += (sign ? amplitude : -amplitude);
			}
		}

		impulse_(0,0) = 0;

		return *this;
	};

	/**
	 * @brief Creates mixed version of filter
	 * @return New mixed filter instance
	 *
	 * @see mix()
	 */
	digital_filter_2d<T, Allocator> mixed() const {
		digital_filter_2d<T, Allocator> ret{*this};

		ret.mix();

		return ret;
	}

	/**
	 * @brief Evaluate digital filter at specific frequency coordinates
	 * @param ux Dimensionless frequency x-component
	 * @param uy Dimensionless frequency y-component
	 * @return Filter response value
	 */
	value_type operator() (value_type ux, value_type uy) const noexcept {
		using std::cos;
		using std::sin;

		const auto& nx = std::get<0>(shape());
		const auto& ny = std::get<1>(shape());

		constexpr auto two_pi = xt::numeric_constants<value_type>::PI * 2;
		const auto cx = cos(two_pi * ux);
		const auto sx = sin(two_pi * ux);
		const auto cy = cos(two_pi * uy);
		const auto sy = sin(two_pi * uy);

		value_type ret = 0;
		value_type six = 0;
		value_type cix = 1;
		for (std::size_t i = 0; i < nx; ++i) {
			const auto i_norm = (i > 0 ? static_cast<value_type>(2) : static_cast<value_type>(1));
			value_type sjy = 0;
			value_type cjy = 1;

			for (std::size_t j = 0; j < ny; ++j) {
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

	/**
	 * @brief Evaluate digital filter for tensor input
	 * @param ex Tensor of dimensionless x-component frequencies
	 * @param ey Tensor of dimensionless y-component frequencies
	 * @return (Nx, Ny) shaped tensor of filter values
	 */
	template<class EX, class EY>
	auto operator() (const xt::xexpression<EX>& ex, const xt::xexpression<EY>& ey) const noexcept {
		return xt::make_lambda_xfunction([this] (const value_type& ux, const value_type& uy) {
			return this->operator()(ux, uy);
		}, xt::expand_dims(ex.derived_cast(), 1), ey.derived_cast());
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

	impulse_type ret{xt::make_lambda_xfunction(std::forward<function_type>(fun), xt::expand_dims(ux, 1), uy)};

	detail::fft_plan_r2r<T> plan{std::array{static_cast<int>(nx), static_cast<int>(ny)},
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

/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_WEIGHT_FUNCTION_GRID_2D_H
#define _WEIF_WEIGHT_FUNCTION_GRID_2D_H

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>

#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xtensor.hpp> // IWYU pragma: keep

#include <weif/detail/fftw3_wrap.h> // IWYU pragma: keep
#include <weif_export.h>

#if __cpp_lib_memory_resource >= 201603
#include <memory_resource>
#endif


namespace weif {
namespace detail {

template<class T>
class WEIF_EXPORT weight_function_grid_2d_base {
public:
	using value_type = T;
	using shape_type = std::array<std::size_t, 2>;

protected:
	using function_type = std::function<value_type(value_type, value_type, value_type)>;

private:
	value_type lambda_;
	value_type aperture_scale_;
	value_type grid_step_;
	shape_type shape_;
	value_type fft_norm_;
	fft_plan_r2r<T> plan_;

protected:
	function_type fun_;

	void apply_inplace_dct(value_type* data) const noexcept { plan_(data, data); }
	const auto& fft_norm() const noexcept { return fft_norm_; }

public:
	weight_function_grid_2d_base(value_type lambda, value_type aperture_scale, value_type grid_step, shape_type shape, function_type&& fun):
		lambda_{lambda},
		aperture_scale_{aperture_scale},
		grid_step_{grid_step},
		shape_{shape},
		fft_norm_{static_cast<value_type>(1) /
			static_cast<value_type>(4 * (std::get<0>(shape_) - 1) * (std::get<1>(shape_) - 1) * grid_step_ * grid_step_)},
		plan_{std::array{static_cast<int>(std::get<0>(shape)), static_cast<int>(std::get<1>(shape))},
			nullptr, nullptr, std::array{FFTW_REDFT00, FFTW_REDFT00}, FFTW_ESTIMATE},
		fun_{std::forward<function_type>(fun)} {}

	const auto& lambda() const noexcept { return lambda_; /* nm */ }
	const auto& aperture_scale() const noexcept { return aperture_scale_; /* mm */ }
	const auto& grid_step() const noexcept { return grid_step_; /* mm */ }
	const auto& shape() const noexcept { return shape_; }
};

} // detail

/**
 * @brief Weight function for uniform grid of identical apertures
 *
 * @tparam T Numeric type for calculations
 * @tparam Allocator Memory allocator type (default: std::allocator<T>)
 *
 * Computes the scintillation weight function for non axially symmetric power spectra:
 * \f[
 * W_{jk}(z) = 9.69 \cdot 10^{-3} \cdot 16 \pi^2 z^{5/6} \lambda^{-7/6} \int \mathbf{du} u^{-8/3} S(u) A\left(\frac{D}{\sqrt{\lambda z}} \mathbf{u}\right) \cos\left(2\pi \frac{\Delta}{\sqrt{\lambda z}} (j \cdot u_x + k \cdot u_y) \right),
 * \f]
 * where \f$ S(u) \f$ is a spectral filter, \f$ \lambda \f$ is its equivalent wavelength, and \f$ A(\mathbf{u}) \f$ is an aperture filter.
 *
 * @par The library uses consistent units:
 * - Altitudes: kilometers (km)
 * - Wavelengths: nanometers (nm)
 * - Geometric scales and grid steps: millimeters (mm)
 */
template<class T, class Allocator = std::allocator<T>>
class WEIF_EXPORT weight_function_grid_2d:
	public detail::weight_function_grid_2d_base<T>,
	private Allocator {
public:
	using typename detail::weight_function_grid_2d_base<T>::value_type; ///< Numeric type for calculations
	using typename detail::weight_function_grid_2d_base<T>::shape_type;
	using allocator_type = Allocator; ///< Memory allocator type
	using result_type = xt::xtensor<value_type, 2, XTENSOR_DEFAULT_LAYOUT, allocator_type>; ///< Result tensor type
	using typename detail::weight_function_grid_2d_base<T>::function_type;

private:
	weight_function_grid_2d(value_type lambda, value_type aperture_scale, value_type grid_step, shape_type shape, function_type&& fun, const allocator_type& alloc):
		detail::weight_function_grid_2d_base<T>(lambda, aperture_scale, grid_step, shape, std::forward<function_type>(fun)),
		allocator_type(alloc) {}

public:
	/**
	 * @brief Construct weight function
	 * @param spectral_filter Spectral filter function
	 * @param lambda Wavelength in nanometers
	 * @param aperture_filter Aperture filter function
	 * @param aperture_scale Aperture scale in millimeters
	 * @param grid_step Grid spacing in millimeters
	 * @param shape Grid dimensions (Nx, Ny)
	 * @param alloc Allocator instance
	 *
	 * The function performs numerical computation each time
	 * weight_function_grid_2d::operator()() is called, returning a tensor
	 * of the specified shape containing the weight function values for the
	 * uniform grid of apertures.
	 *
	 * @see operator()()
	 */
	template<class SF, class AF>
	weight_function_grid_2d(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, value_type grid_step, shape_type shape, const allocator_type& alloc = allocator_type()):
		weight_function_grid_2d(lambda, aperture_scale, grid_step, shape,
			[spectral_filter = std::forward<SF>(spectral_filter), aperture_filter = std::forward<AF>(aperture_filter)] (value_type ux, value_type uy, value_type x) noexcept -> value_type {
			if (ux == static_cast<value_type>(0) && uy == static_cast<value_type>(0))
				return static_cast<value_type>(0);

			if (ux == std::numeric_limits<value_type>::infinity() || uy == std::numeric_limits<value_type>::infinity())
				return static_cast<value_type>(0);

			const auto u2 = ux * ux + uy * uy;

			if (u2 < static_cast<value_type>(1)) {
				return std::pow(u2, static_cast<value_type>(1.0/6.0)) * spectral_filter.regular(u2) * aperture_filter(x * ux, x * uy);
			}

			return std::pow(u2, -static_cast<value_type>(11.0/6.0)) * spectral_filter(u2) * aperture_filter(x * ux, x * uy);
		}, alloc) {}

	template<class SF, class AF>
	weight_function_grid_2d(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, shape_type shape, const allocator_type& alloc = allocator_type()):
		weight_function_grid_2d(std::forward<SF>(spectral_filter), lambda, std::forward<AF>(aperture_filter), aperture_scale, aperture_scale, shape, alloc) {}

	/// @return Associated allocator
	const allocator_type& get_allocator() const noexcept { return *this; }

	/**
	 * @brief Evaluate weight function for uniform aperture grid at specific altitude
	 * @param altitude Atmospheric altitude in kilometers
	 * @return 2D tensor of weight function values on spatial grid of shape (Nx, Ny)
	 */
	inline result_type operator() (value_type altitude) const {
		using namespace std;
		using namespace std::placeholders;

		if (altitude == static_cast<value_type>(0)) {
			return xt::zeros<value_type>(this->shape());
		}

		constexpr const auto PI = xt::numeric_constants<value_type>::PI;
		/* 1e13 = pow(1e3, 5.0/6.0) * pow(1e9, 7.0/6.0) */
		constexpr const value_type c = 9.69e-3 * 16 * PI * PI * 1e13;

		const value_type fresnel_radius = sqrt(this->lambda() * altitude);
		const value_type nyquist = fresnel_radius / this->grid_step() / 2;

		const auto ux = xt::linspace(static_cast<value_type>(0), nyquist, std::get<0>(this->shape()));
		const auto uy = xt::linspace(static_cast<value_type>(0), nyquist, std::get<1>(this->shape()));

		result_type res{xt::make_lambda_xfunction(
			std::bind(std::cref(detail::weight_function_grid_2d_base<T>::fun_), _1, _2, this->aperture_scale() / fresnel_radius),
			xt::expand_dims(ux, 1), uy)};

		this->apply_inplace_dct(res.data());

		res *= c * this->fft_norm() / pow(this->lambda(), static_cast<value_type>(1.0/6.0)) * pow(altitude, static_cast<value_type>(11.0/6.0));

		return res;
	}
};


extern template class weight_function_grid_2d<float>;
extern template class weight_function_grid_2d<double>;
extern template class weight_function_grid_2d<long double>;

#if __cpp_lib_memory_resource >= 201603

namespace pmr {

template<class T>
using weight_function_grid_2d = weif::weight_function_grid_2d<T, std::pmr::polymorphic_allocator<T>>;

} // pmr

extern template class weight_function_grid_2d<float, std::pmr::polymorphic_allocator<float>>;
extern template class weight_function_grid_2d<double, std::pmr::polymorphic_allocator<double>>;
extern template class weight_function_grid_2d<long double, std::pmr::polymorphic_allocator<long double>>;

#endif

} // weif

#endif // _WEIF_WEIGHT_FUNCTION_GRID_H

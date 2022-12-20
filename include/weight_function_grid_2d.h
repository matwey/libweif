#ifndef _WEIF_WEIGHT_FUNCTION_GRID_2D_H
#define _WEIF_WEIGHT_FUNCTION_GRID_2D_H

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>

#include <fftw3_wrap.h>

#include <weif_export.h>


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

template<class T, class Allocator = std::allocator<T>>
class WEIF_EXPORT weight_function_grid_2d:
	public detail::weight_function_grid_2d_base<T>,
	private Allocator {
public:
	using typename detail::weight_function_grid_2d_base<T>::value_type;
	using typename detail::weight_function_grid_2d_base<T>::shape_type;
	using allocator_type = Allocator;
	using result_type = xt::xtensor<value_type, 2, XTENSOR_DEFAULT_LAYOUT, allocator_type>;
	using typename detail::weight_function_grid_2d_base<T>::function_type;

private:
	weight_function_grid_2d(value_type lambda, value_type aperture_scale, value_type grid_step, shape_type shape, function_type&& fun, const allocator_type& alloc):
		detail::weight_function_grid_2d_base<T>(lambda, aperture_scale, grid_step, shape, std::forward<function_type>(fun)),
		allocator_type(alloc) {}

public:
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

	const allocator_type& get_allocator() const noexcept { return *this; }

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
			ux, xt::expand_dims(uy, 1))};

		this->apply_inplace_dct(res.data());

		res *= c * this->fft_norm() / pow(this->lambda(), static_cast<value_type>(1.0/6.0)) * pow(altitude, static_cast<value_type>(11.0/6.0));

		return res;
	}
};


extern template class weight_function_grid_2d<float>;
extern template class weight_function_grid_2d<double>;
extern template class weight_function_grid_2d<long double>;

namespace pmr {

template<class T>
using weight_function_grid_2d = weif::weight_function_grid_2d<T, std::pmr::polymorphic_allocator<T>>;

} // pmr

extern template class weight_function_grid_2d<float, std::pmr::polymorphic_allocator<float>>;
extern template class weight_function_grid_2d<double, std::pmr::polymorphic_allocator<double>>;
extern template class weight_function_grid_2d<long double, std::pmr::polymorphic_allocator<long double>>;

} // weif

#endif // _WEIF_WEIGHT_FUNCTION_GRID_H

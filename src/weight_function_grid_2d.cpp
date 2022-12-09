#include <weight_function_grid_2d.h>


namespace weif {

template<class T>
weight_function_grid_2d_base<T>::weight_function_grid_2d_base(value_type lambda, value_type aperture_scale, value_type grid_step, shape_type shape, function_type&& fun):
	lambda_{lambda},
	aperture_scale_{aperture_scale},
	grid_step_{grid_step},
	shape_{shape},
	fft_norm_{static_cast<value_type>(1) /
		static_cast<value_type>(4 * (std::get<0>(shape_) - 1) * (std::get<1>(shape_) - 1) * grid_step_ * grid_step_)},
	plan_{std::array{static_cast<int>(std::get<0>(shape)), static_cast<int>(std::get<1>(shape))},
		nullptr, nullptr, std::array{FFTW_REDFT00, FFTW_REDFT00}, FFTW_ESTIMATE},
	fun_{std::forward<function_type>(fun)} {}

template<class T>
void weight_function_grid_2d_base<T>::apply_inplace_dct(value_type* data) const noexcept {
	plan_(data, data);
}

template class weight_function_grid_2d_base<float>;
template class weight_function_grid_2d_base<double>;
template class weight_function_grid_2d_base<long double>;

} // weif

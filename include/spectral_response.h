#ifndef _WEIF_SPECTRAL_RESPONSE_H
#define _WEIF_SPECTRAL_RESPONSE_H

#include <string>

#include <xtensor/xtensor.hpp>

#include <uniform_grid.h>

#include <weif_export.h>


namespace weif {

template<class T>
class WEIF_EXPORT spectral_response {
public:
	static_assert(std::is_floating_point<T>::value, "type T is not supported");

	using value_type = T;
private:
	/* Consider using different type for uniform_grid, for instance, boost::cpp_dec_float */
	uniform_grid<value_type>  grid_;
	xt::xtensor<value_type, 1> data_;
public:
	spectral_response(const uniform_grid<value_type>& grid, const xt::xtensor<value_type, 1>& data):
		grid_{grid},
		data_{data} {}

	const auto& grid() const { return grid_; }
	const auto& data() const { return data_; }

	void normalize();
	spectral_response<value_type> normalized() const;

	void stack(const spectral_response<value_type>& other);
	spectral_response<value_type> stacked(const spectral_response<value_type>& other) const;

	static spectral_response<value_type> make_from_file(const std::string& filename);
};

template<class T>
spectral_response(const uniform_grid<T>& grid, const xt::xtensor<T, 1>& data) -> spectral_response<T>;

} // weif

#endif // _WEIF_SPECTRAL_RESPONSE_H

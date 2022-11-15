#ifndef _WEIF_SPECTRAL_FILTER_H
#define _WEIF_SPECTRAL_FILTER_H

#include <string>

#include <xtensor/xtensor.hpp>

#include <spectral_response.h>
#include <uniform_grid.h>

#include <weif_export.h>


namespace weif {

template<class T>
class WEIF_EXPORT spectral_filter {
public:
//	static_assert(std::is_floating_point<T>::value, "type T is not supported");
	static_assert(std::is_same_v<T, float>, "type T is not supported");

	using value_type = T;
private:
	/* Consider using different type for uniform_grid, for instance, boost::cpp_dec_float */
	uniform_grid<value_type>   grid_;
	xt::xtensor<value_type, 1> data_;
public:
	spectral_filter(const spectral_response<value_type>& response, std::size_t size);
	spectral_filter(const uniform_grid<value_type>& grid, const xt::xtensor<value_type, 1>& data):
		grid_{grid},
		data_{data} {}

	const auto& grid() const { return grid_; }
	const auto& data() const { return data_; }

	value_type equiv_lambda() const noexcept;

	void dump(const std::string& filename) const;
};

template<class T>
spectral_filter(const spectral_response<T>& response) -> spectral_filter<T>;
template<class T>
spectral_filter(const uniform_grid<T>& grid, const xt::xtensor<T, 1>& data) -> spectral_filter<T>;

} // weif

#endif // _WEIF_SPECTRAL_FILTER_H

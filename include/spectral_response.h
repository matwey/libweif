#ifndef _WEIF_SPECTRAL_RESPONSE_H
#define _WEIF_SPECTRAL_RESPONSE_H

#include <algorithm>
#include <string>
#include <optional>

#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <rapidcsv.h>

#include <uniform_grid.h>

#include <weif_export.h>


namespace weif {

template<class T>
class WEIF_EXPORT spectral_response {
public:
	using value_type = T;

private:
	/* Consider using different type for uniform_grid, for instance, boost::cpp_dec_float */
	uniform_grid<value_type>   grid_;
	xt::xtensor<value_type, 1> data_;

public:
	spectral_response(const uniform_grid<value_type>& grid, const xt::xtensor<value_type, 1>& data):
		grid_{grid},
		data_{data} {}

	const auto& grid() const { return grid_; }
	const auto& data() const { return data_; }

	void normalize() noexcept;
	spectral_response<value_type> normalized() const;

	void stack(const spectral_response<value_type>& other);
	spectral_response<value_type> stacked(const spectral_response<value_type>& other) const;

	value_type effective_lambda() const noexcept {
		return grid_.origin() + grid_.delta() * xt::average(xt::arange(data_.size()), data_ / grid_.values())();
	}

	static spectral_response<value_type> make_from_file(const std::string& filename);

	template<class Iterator>
	static spectral_response<value_type> stack_from_files(Iterator begin, Iterator end);
};

template<class T>
void spectral_response<T>::normalize() noexcept {
	/* Response in full bandwidth is 1 */
	const auto norm = xt::sum(data_)();

	data_ /= norm;
}

template<class T>
spectral_response<T> spectral_response<T>::normalized() const {
	spectral_response<T> ret{*this};

	ret.normalize();

	return ret;
}

template<class T>
void spectral_response<T>::stack(const spectral_response<value_type>& other) {
	const auto i_grid     = grid_.intersect(other.grid());
	const auto idx        = grid_.to_index(i_grid.origin());
	const auto other_idx  = other.grid().to_index(i_grid.origin());
	const auto view       = xt::view(data_,        xt::range(idx, idx + i_grid.size()));
	const auto other_view = xt::view(other.data(), xt::range(other_idx, other_idx + i_grid.size()));

	assert(view.size() == other_view.size());

	data_ = view * other_view;
	grid_ = i_grid;
}

template<class T>
spectral_response<T> spectral_response<T>::stacked(const spectral_response<value_type>& other) const {
	spectral_response<T> ret{*this};

	ret.stack(other);

	return ret;
}

template<class T>
spectral_response<T> spectral_response<T>::make_from_file(const std::string& filename) {
	rapidcsv::Document doc(filename,
		rapidcsv::LabelParams(-1, -1),
		rapidcsv::SeparatorParams(' ', true),
		rapidcsv::ConverterParams(),
		rapidcsv::LineReaderParams(true));

	const auto grid_column = doc.GetColumn<T>(0);
	const auto data_column = doc.GetColumn<T>(1);
	const auto data_adapt  = xt::adapt(data_column, {data_column.size()});

	return {uniform_grid{grid_column.cbegin(), grid_column.cend()}, data_adapt};
}

template<class T>
template<class Iterator>
spectral_response<T> spectral_response<T>::stack_from_files(Iterator begin, Iterator end) {
	using optional_type = std::optional<spectral_response<T>>;

	return std::accumulate(begin, end, optional_type{}, [] (const optional_type& acc, const auto& filename) {
		auto c = spectral_response<T>::make_from_file(filename);

		if (acc) {
			c.stack(*acc);
		}

		return c;
	}).value();
}

template<class T>
spectral_response(const uniform_grid<T>& grid, const xt::xtensor<T, 1>& data) -> spectral_response<T>;


extern template class spectral_response<float>;
extern template class spectral_response<double>;
extern template class spectral_response<long double>;

} // weif

#endif // _WEIF_SPECTRAL_RESPONSE_H

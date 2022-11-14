#include <rapidcsv.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <spectral_response.h>


namespace weif {

template<class T>
void spectral_response<T>::normalize() {
	/* Response in full bandwidth is 1 */
	const auto norm = grid().delta() * (
		xt::sum(data_)() - (*data_.cbegin() + *data_.crbegin()) / static_cast<T>(2));

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

template class spectral_response<float>;
template class spectral_response<double>;
template class spectral_response<long double>;

} // weif

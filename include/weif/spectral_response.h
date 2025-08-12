/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_SPECTRAL_RESPONSE_H
#define _WEIF_SPECTRAL_RESPONSE_H

#include <algorithm>
#include <cassert>
#include <numeric>
#include <optional>
#include <string>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include <rapidcsv.h>

#include <weif/uniform_grid.h>
#include <weif_export.h>


namespace weif {

/**
 * @brief Spectral response
 *
 * @tparam T Numeric type used for calculations
 *
 * Represents spectral response curve on a uniform wavelength grid,
 * providing operations for normalization, stacking, and analysis.
 * The spectral response is typically used as input for polychromatic
 * filter calculations.
 *
 * @see sf::poly
 */
template<class T>
class WEIF_EXPORT spectral_response {
public:
	using value_type = T; ///< Numeric type for spectral calculations

private:
	/* Consider using different type for uniform_grid, for instance, boost::cpp_dec_float */
	uniform_grid<value_type>   grid_; ///< Wavelength grid definition
	xt::xtensor<value_type, 1> data_; ///< Spectral response values

public:
	/**
	 * @brief Construct from grid and data
	 * @param grid Wavelength grid definition
	 * @param data Spectral response curve values
	 */
	spectral_response(const uniform_grid<value_type>& grid, const xt::xtensor<value_type, 1>& data):
		grid_{grid},
		data_{data} {}

	/// @return Reference to the wavelength grid
	const auto& grid() const { return grid_; }
	/// @return Reference to the spectral response surve
	const auto& data() const { return data_; }

	/**
	 * @brief Normalizes the spectral response in-place
	 *
	 * Scales the data so that the total response equals 1:
	 * \f[
	 * \sum_i F(\lambda_i) = 1.
	 * \f]
	 *
	 * @return Reference to modified object
	 */
	spectral_response<value_type>& normalize() noexcept;

	/**
	 * @brief Creates normalized copy of the response
	 * @see normalize()
	 * @return New normalized spectral response
	 */
	spectral_response<value_type> normalized() const;


	/**
	 * @brief Performs in-place spectral response stacking (multiplication)
	 *
	 * @param other Another spectral response to stack with current
	 *
	 * @par The stacking operation:
	 * - Finds the wavelength range common to both responses (intersection)
	 * - Extracts corresponding data segments from both responses
	 * - Computes element-wise product of the response values
	 * - Updates the object to contain only the stacked intersection range
	 *
	 * Both spectral responses must have compatible wavelength grids: sufficient overlapping wavelength range and identical spacing.
	 *
	 * @throws mismatched_grids If the grids have no overlapping wavelength range
	 *
	 * @see stacked()
	 */
	void stack(const spectral_response<value_type>& other);

	/**
	 * @brief Creates stacked response (element-wise multiplication)
	 * @see stack()
	 * @param other Response to stack with current
	 * @return New stacked spectral response
	 */
	spectral_response<value_type> stacked(const spectral_response<value_type>& other) const;

	/// @return Effective wavelength
	value_type effective_lambda() const noexcept {
		return grid_.origin() + grid_.delta() * xt::average(xt::arange(data_.size()), data_ / grid_.values())();
	}

	/**
	 * @brief Loads spectral response from file
	 * @param filename File containing wavelength and response columns
	 * @par File Format Requirements:
	 * - Space-separated values (no multiple spaces are allowed as delimiter)
	 * - No header row
	 * - First column: Wavelength values (in units of nm., increasing order)
	 * - Second column: Corresponding spectral response values
	 * @par Example valid file content:
	 *   @code
	 *   400.0 0.15
	 *   410.0 0.25
	 *   ...
	 *   700.0 0.05
	 *   @endcode
	 * @return New spectral response instance
	 */
	static spectral_response<value_type> make_from_file(const std::string& filename);

	/**
	 * @brief Creates stacked response from multiple files
	 * @tparam Iterator Input iterator type
	 * @param begin Start iterator for filenames
	 * @param end End iterator for filenames
	 * @see make_from_file()
	 * @return New stacked spectral response
	 */
	template<class Iterator>
	static spectral_response<value_type> stack_from_files(Iterator begin, Iterator end);
};

template<class T>
spectral_response<T>& spectral_response<T>::normalize() noexcept {
	/* Response in full bandwidth is 1 */
	const auto norm = xt::sum(data_)();

	data_ /= norm;

	return *this;
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

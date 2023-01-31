/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_DETAIL_UNIFORM_GRID_H
#define _WEIF_DETAIL_UNIFORM_GRID_H

#include <cstdlib>
#include <type_traits>

#include <weif/error.h>


namespace weif {
namespace detail {

template<class T>
struct uniform_grid_fn {
	using value_type = T;

	value_type origin;
	value_type delta;

	value_type operator() (std::size_t ind) const {
		return origin + ind * delta;
	}

	template<class Iter>
	value_type element(Iter it, Iter) const {
		return operator() (*it);
	}

	template<class Iter,
		std::enable_if_t<
			std::is_convertible<
				typename std::iterator_traits<Iter>::value_type, T>::value, bool> = true>
	static uniform_grid_fn<T> make_from_iterable(Iter begin, Iter end) {
		uniform_grid_fn<T> grid_fn{0, 1};

		if (begin == end)
			return grid_fn;

		grid_fn.origin = *begin++;

		if (begin == end)
			return grid_fn;

		grid_fn.delta = *begin++ - grid_fn.origin;

		for (std::size_t i = 2; begin != end; ++begin, ++i) {
			if (grid_fn(i) != *begin)
				throw non_uniform_grid{i, *begin, grid_fn(i)};
		}

		return grid_fn;
	}
};

} // detail
} // weif

#endif // _WEIF_DETAIL_UNIFORM_GRID_H

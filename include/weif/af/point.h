/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_POINT_H
#define _WEIF_AF_POINT_H

#include <type_traits>

#include <weif_export.h>


namespace weif {
namespace af {

template<class T>
struct WEIF_EXPORT point {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		return static_cast<value_type>(1);
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return static_cast<value_type>(1);
	}

	template<class E, xt::enable_xexpression<std::decay_t<E>, bool> = true>
	auto operator() (E&& e) const noexcept {
		return xt::ones_like(std::forward<E>(e));
	}

	template<class E1, class E2, xt::enable_xexpression<std::decay_t<E1>, bool> = true, xt::enable_xexpression<std::decay_t<E2>, bool> = true>
	auto operator() (E1&& e1, E2&& e2) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<E1>(e1), std::forward<E2>(e2));

		return xt::ones_like(std::move(xx));
	}
};

} // af
} // weif

#endif // _WEIF_AF_POINT_H

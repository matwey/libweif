/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_GAUSS_H
#define _WEIF_AF_GAUSS_H

#include <cmath>

#include <xtensor/xmath.hpp>
#include <xtensor/xutils.hpp>
#include <xtensor/xvectorize.hpp>

#include <weif_export.h>


namespace weif {
namespace af {

template<class T>
struct WEIF_EXPORT gauss {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		return std::exp(-u*u);
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return std::exp(-ux*ux-uy*uy);
	}

	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		return xt::exp(-xt::square(std::forward<E>(e)));
	}

	template<class E1, class E2, xt::enable_xexpression<E1, bool> = true, xt::enable_xexpression<E2, bool> = true>
	auto operator() (E1&& e1, E2&& e2) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<E1>(e1), std::forward<E2>(e2));

		return xt::exp(-xt::square(std::move(xx)) - xt::square(std::move(yy)));
	}
};

} // af
} // weif

#endif // _WEIF_AF_GAUSS_H


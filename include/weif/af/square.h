/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_SQUARE_H
#define _WEIF_AF_SQUARE_H

#include <cmath>

#include <xtensor/xmath.hpp>
#include <xtensor/xvectorize.hpp>

#include <weif/math.h>
#include <weif_export.h>


namespace weif {
namespace af {

template<class T>
struct WEIF_EXPORT square {
	using value_type = T;

	value_type operator() (value_type ux, value_type uy) const noexcept {
		using namespace std;
		using boost::math::sinc_pi;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(sinc_pi(ux * PI) * sinc_pi(uy * PI), 2);
	}

	template<class E1, class E2, xt::enable_xexpression<E1, bool> = true, xt::enable_xexpression<E2, bool> = true>
	auto operator() (E1&& e1, E2&& e2) const noexcept {
		auto [xx, yy] = xt::meshgrid(
			math::sinc_pi(xt::numeric_constants<value_type>::PI * std::forward<E1>(e1)),
			math::sinc_pi(xt::numeric_constants<value_type>::PI * std::forward<E2>(e2)));

		return xt::square(std::move(xx) * std::move(yy));
	}
};

} // af
} // weif

#endif // _WEIF_AF_SQUARE_H

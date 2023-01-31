/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_SQUARE_H
#define _WEIF_AF_SQUARE_H

#include <cmath>

#include <boost/math/special_functions/sinc.hpp>

#include <xtensor/xmath.hpp>
#include <xtensor/xvectorize.hpp>

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

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E1>& e2) const noexcept {
		using boost::math::sinc_pi;

		const auto sinc_pi_vec = xt::vectorize(&sinc_pi<value_type>);

		return xt::square(sinc_pi_vec(xt::numeric_constants<value_type>::PI * e1.derived_cast()) * sinc_pi_vec(xt::numeric_constants<value_type>::PI * xt::expand_dims(e2.derived_cast(), 1)));
	}
};

} // af
} // weif

#endif // _WEIF_AF_SQUARE_H

/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_SF_MONO_H
#define _WEIF_SF_MONO_H

#include <cmath>
#include <type_traits>

#include <xtensor/xmath.hpp>

#include <weif/math.h>
#include <weif_export.h>


namespace weif {
namespace sf {

template<class T>
class WEIF_EXPORT mono {
public:
	using value_type = T;

	value_type operator() (const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(sin(PI * x), 2);
	}

	value_type regular(const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;
		using boost::math::sinc_pi;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(PI * sinc_pi(PI * x), 2);
	}

	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return xt::square(xt::sin(static_cast<xvalue_type>(PI) * std::forward<E>(e)));
	}

	template<class E, xt::enable_xexpression<E, bool> = true>
	auto regular(E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return xt::square(static_cast<xvalue_type>(PI) * math::sinc_pi(static_cast<xvalue_type>(PI) * std::forward<E>(e)));
	}
};

} // sf
} // weif

#endif // _WEIF_SF_MONO_H

/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2024  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_MATH_H
#define _WEIF_MATH_H

#include <cmath>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/math/tools/precision.hpp>

#include <xtensor/xmath.hpp>


namespace weif {
namespace math {

template<class T, xt::disable_xexpression<T, bool> = true>
T jinc_pi(const T x) noexcept {
	using namespace std;

	if (abs(x) >= 3.7 * boost::math::tools::forth_root_epsilon<T>()) {
		return boost::math::cyl_bessel_j(1, x) / x * static_cast<T>(2);
	} else {
        	// |x| < (eps*192)^(1/4)
		return static_cast<T>(1) - x * x / 8;
	}
}

template<class T, xt::disable_xexpression<T, bool> = true>
T zinc_pi(const T x) noexcept {
	using namespace std;

	if (abs(x) >= 7.2 * boost::math::tools::forth_root_epsilon<T>()) {
		return boost::math::cyl_bessel_j(2, x) / (x * x) * static_cast<T>(8);
	} else {
        	// |x| < (eps*384)^(1/4)
		return static_cast<T>(1) - x * x / 12;
	}
}

namespace detail {

	struct jinc_pi_fun {
		template <class T>
		constexpr auto operator()(const T& arg) const {
			return jinc_pi(arg);
		}

		template <class B>
		constexpr auto simd_apply(const B& arg) const {
			return jinc_pi(arg);
		}
	};

	struct sinc_pi_fun {
		template <class T>
		constexpr auto operator()(const T& arg) const {
			using boost::math::sinc_pi;
			return sinc_pi(arg);
		}

		template <class B>
		constexpr auto simd_apply(const B& arg) const {
			using boost::math::sinc_pi;
			return sinc_pi(arg);
		}
	};

	struct zinc_pi_fun {
		template <class T>
		constexpr auto operator()(const T& arg) const {
			return zinc_pi(arg);
		}

		template <class B>
		constexpr auto simd_apply(const B& arg) const {
			return zinc_pi(arg);
		}
	};

} // detail

template <class E, xt::enable_xexpression<E, bool> = true>
inline auto jinc_pi(E&& e) noexcept {
	return xt::xfunction<detail::jinc_pi_fun, E>{detail::jinc_pi_fun{}, std::forward<E>(e)};
}

template <class E, xt::enable_xexpression<E, bool> = true>
inline auto sinc_pi(E&& e) noexcept {
	return xt::xfunction<detail::sinc_pi_fun, E>{detail::sinc_pi_fun{}, std::forward<E>(e)};
}

template <class E, xt::enable_xexpression<E, bool> = true>
inline auto zinc_pi(E&& e) noexcept {
	return xt::xfunction<detail::zinc_pi_fun, E>{detail::zinc_pi_fun{}, std::forward<E>(e)};
}

} // math
} // weif

#endif // _WEIF_MATH_H

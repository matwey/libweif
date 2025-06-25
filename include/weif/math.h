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

/**
 * @defgroup math_functions Special mathematical functions
 * @brief Special mathematical functions
 */

/// @namespace math
/// @brief Mathematical constants and functions namespace
namespace math {

/**
 * @brief Kolmogorov turbulence structure constant scaling factor
 *
 * @tparam T Numeric type for the constant
 *
 * Represents the fundamental scaling factor in Kolmogorov turbulence theory:
 * \f[
 * C = \frac{\Gamma\left(\frac{8}{3}\right) \sin\left(\frac{\pi}{3}\right)}{(2\pi)^{8/3}} \approx 9.69 \cdot 10^{-3},
 * \f]
 * where:
 * - \f$\Gamma\f$ is the Gamma function
 * - The value is precomputed to maximum available precision
 */
template<class T>
constexpr T Kolmogorov_Cn2_scale = T(0.0096931507043123421456817216188956817L);


/**
 * @brief Computes the jinc function
 * @ingroup math_functions
 *
 * @tparam T Numeric type
 * @param x Input value
 * @return Value of the jinc function at x
 *
 * The function provides accurate computation across the full numeric range:
 * \f$ \mathrm{jinc}(x) = \frac{2 \cdot J_1(x)}{x}. \f$
 *
 * @note Uses Boost.Math for Bessel function evaluation
 */
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

} // detail

template <class E, xt::enable_xexpression<E, bool> = true>
inline auto jinc_pi(E&& e) noexcept {
	return xt::xfunction<detail::jinc_pi_fun, E>{detail::jinc_pi_fun{}, std::forward<E>(e)};
}

template <class E, xt::enable_xexpression<E, bool> = true>
inline auto sinc_pi(E&& e) noexcept {
	return xt::xfunction<detail::sinc_pi_fun, E>{detail::sinc_pi_fun{}, std::forward<E>(e)};
}

} // math
} // weif

#endif // _WEIF_MATH_H

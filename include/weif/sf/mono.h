/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_SF_MONO_H
#define _WEIF_SF_MONO_H

#include <cmath>
#include <type_traits>

#include <xtensor/core/xmath.hpp>

#include <weif/math.h>
#include <weif_export.h>


namespace weif {
namespace sf {

/**
 * @brief Monocrhomatic spectral filter
 *
 * @tparam T Numeric type used for calculations
 *
 * The monochromatic spectral filter is defined as:
 * \f[
 * E(x) = \sin^2(\pi x),
 * \f]
 * where \f$x \equiv z f^2 = \frac{u^2}{\lambda}\f$.
 */
template<class T>
class WEIF_EXPORT mono {
public:
	using value_type = T; ///< Numeric type used for calculations

	/**
	 * @brief Call operator for monochromatic spectral filter
	 *
	 * Evaluates the filter function for a given argument:
	 * \f$ E(x) = \sin^2(\pi x). \f$
	 *
	 * @param x Normalized squared frequency \f$x = z f^2 = \frac{u^2}{\lambda}\f$
	 * @return Filter value at x
	 */
	value_type operator() (const value_type x) const noexcept {
		using namespace std;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(sin(PI * x), 2);
	}

	/**
	 * @brief Evaluate regularized monochromatic spectral filter
	 *
	 * Evaluates \f$ x^2 E(x). \f$
	 *
	 * @param x Normalized squared frequency \f$x = z f^2 = \frac{u^2}{\lambda}\f$
	 * @return Regularized filter value at x
	 */
	value_type regular(const value_type x) const noexcept {
		using namespace std;
		using boost::math::sinc_pi;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(PI * sinc_pi(PI * x), 2);
	}

        /**
         * @brief Call operator for monochromatic spectral filter with tensor input
         *
         * Evaluates the filter function for an array of arguments:
         * \f$ E(x) = \sin^2(\pi x). \f$
         *
         * @param e Input tensor of dimensionless spatial frequency magnitudes
         * @return Tensor of spectral filter values with same shape as input
         */
	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		constexpr auto PI = xt::numeric_constants<xvalue_type>::PI;

		return xt::square(xt::sin(static_cast<xvalue_type>(PI) * std::forward<E>(e)));
	}

	/**
	 * @brief Evaluate regularized monochromatic spectral filter for tensor input
	 *
	 * Evaluates \f$ x^2 E(x) \f$ for an array of arguments.
	 *
	 * @param e Input tensor of dimensionless spatial frequency magnitudes
	 * @return Tensor of regularized spectral filter values with same shape as input
	 */
	template<class E, xt::enable_xexpression<E, bool> = true>
	auto regular(E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		constexpr auto PI = xt::numeric_constants<xvalue_type>::PI;

		return xt::square(static_cast<xvalue_type>(PI) * math::sinc_pi(static_cast<xvalue_type>(PI) * std::forward<E>(e)));
	}
};

} // sf
} // weif

#endif // _WEIF_SF_MONO_H

/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_SF_GAUSS_H
#define _WEIF_SF_GAUSS_H

#include <cmath>

#include <xtensor/core/xmath.hpp>
#include <xtensor/utils/xutils.hpp>
#include <xtensor/core/xvectorize.hpp>

#include <weif/math.h>
#include <weif_export.h>


namespace weif {
namespace sf {

/**
 * @brief Gaussian spectral filter
 *
 * @tparam T Numeric type used for calculations
 *
 * The filter combines monochromatic oscillations with Gaussian damping:
 * \f[
 * E(x) = \sin^2(\pi x) \cdot \exp\left(-\frac{\pi^2}{8\ln 2} (x \Lambda)^2\right)
 * \f]
 * where:
 * - \f$ x \equiv z f^2 = \frac{u^2}{\lambda} \f$,
 * - \f$ \Lambda \f$ is the full width at half maximum of the Gaussian envelope expressed in relative units.
 *
 * Reference: Tokovinin (2003) "Polychromatic scintillation", https://doi.org/10.1364/JOSAA.20.000686
 */
template<class T>
class WEIF_EXPORT gauss {
public:
	using value_type = T; ///< Numeric type used for calculations

private:
	value_type fwhm_;

public:
	/**
	 * @brief Construct a Gaussian spectral filter
	 *
	 * @param fwhm Full width at half maximum \f$\Lambda\f$ of the Gaussian envelope expressed in relative units
	 */
	explicit gauss(value_type fwhm) noexcept:
		fwhm_{fwhm} {}

	/// @brief Returns full width at half maximum parameter \f$\Lambda\f$
	/// @return Full width at half maximum parameter \f$\Lambda\f$
	value_type fwhm() const noexcept { return fwhm_; }

	/**
	 * @brief Call operator for Gaussian spectral filter
	 *
	 * Evaluates the filter function for a given argument:
	 * \f[
	 * E(x) = \sin^2(\pi x) \cdot \exp\left(-\frac{\pi^2}{8\ln 2} (x \Lambda)^2\right)
	 * \f]
	 *
	 * @param x Normalized squared frequency \f$x = z f^2 = \frac{u^2}{\lambda}\f$
	 * @return Filter value at x
	 */
	value_type operator() (const value_type x) const noexcept {
		using namespace std;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;
		constexpr auto C = static_cast<value_type>(1) / xt::numeric_constants<value_type>::LN2 / 8;
		const auto pix = PI * x;

		value_type e = exp(-C * pow(fwhm() * pix, 2));
		if (e == static_cast<value_type>(0))
			return 0;

		return e * pow(sin(pix), 2);
	}

	/**
	 * @brief Evaluate regularized Gaussian spectral filter
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
		constexpr auto C = static_cast<value_type>(1) / xt::numeric_constants<value_type>::LN2 / 8;
		const auto pix = PI * x;

		return pow(PI * sinc_pi(pix), 2) * exp(-C * pow(fwhm() * pix, 2));
	}

        /**
         * @brief Call operator for Gaussian spectral filter with tensor input
         *
         * Evaluates the filter function for an array of arguments:
	 * \f[
	 * E(x) = \sin^2(\pi x) \cdot \exp\left(-\frac{\pi^2}{8\ln 2} (x \Lambda)^2\right)
	 * \f]
         *
         * @param e Input tensor of dimensionless spatial frequency magnitudes
         * @return Tensor of spectral filter values with same shape as input
         */
	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		const auto fwhm_ = fwhm();

		auto fnct = [fwhm_](auto x) -> decltype(x) {
			using namespace std;

			constexpr auto PI = xt::numeric_constants<xvalue_type>::PI;
			constexpr auto C = static_cast<value_type>(1) / xt::numeric_constants<value_type>::LN2 / 8;
			const auto pix = PI * x;

			value_type e = exp(-C * pow(fwhm_ * pix, 2));
			if (e == static_cast<value_type>(0))
				return 0;

			return e * pow(sin(pix), 2);
		};

		return xt::make_lambda_xfunction(std::move(fnct), std::forward<E>(e));
	}

	/**
	 * @brief Evaluate regularized Gaussian spectral filter for tensor input
	 *
	 * Evaluates \f$ x^2 E(x) \f$ for an array of arguments.
	 *
	 * @param e Input tensor of dimensionless spatial frequency magnitudes
	 * @return Tensor of regularized spectral filter values with same shape as input
	 */
	template<class E, xt::enable_xexpression<E, bool> = true>
	auto regular(E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		const auto fwhm_ = fwhm();

		auto fnct = [fwhm_](auto x) -> decltype(x) {
			using namespace std;
			using boost::math::sinc_pi;

			constexpr auto PI = xt::numeric_constants<xvalue_type>::PI;
			constexpr auto C = static_cast<value_type>(1) / xt::numeric_constants<value_type>::LN2 / 8;
			const auto pix = PI * x;

			return pow(PI * sinc_pi(pix), 2) * exp(-C * pow(fwhm_ * pix, 2));
		};

		return xt::make_lambda_xfunction(std::move(fnct), std::forward<E>(e));
	}
};

} // sf
} // weif

#endif // _WEIF_SF_GAUSS_H


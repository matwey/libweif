/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_CIRCULAR_H
#define _WEIF_AF_CIRCULAR_H

#include <cmath>
#include <type_traits>

#include <xtensor/xmath.hpp>
#include <xtensor/xutils.hpp>
#include <xtensor/xvectorize.hpp>

#include <weif/math.h>
#include <weif_export.h>


namespace weif {
namespace af {

/**
 * @brief Aperture filter function for a circular aperture
 *
 * @tparam T Numeric type used for calculations.
 *
 * The aperture filter is defined in both radial and Cartesian plane coordinates:
 * \f[
 * A(u) = \mathrm{jinc}_1^2(\pi u),
 * \f]
 * \f[
 * A(u_x, u_y) = \mathrm{jinc}_1^2\left(\pi \sqrt{u_x^2 + u_y^2}\right).
 * \f]
 * where \f$\mathrm{jinc}_1(x) = \frac{2 J_1(x)}{x}\f$ is the jinc function (Fourier transform of
 * a unit circular aperture) and \f$J_1\f$ is the Bessel function of first kind.
 */
template<class T>
struct WEIF_EXPORT circular {
	using value_type = T; ///< Numeric type used for calculations

	/**
	 * @brief Call operator for circular aperture filter in radial coordinates
	 *
	 * Evaluates the squared jinc function for given radial frequency:
	 * \f$ A(u) = \mathrm{jinc}_1^2(\pi u) \f$.
	 *
	 * @param u Dimensionless spatial frequency magnitude (radial coordinate)
	 * @return Aperture filter value at specified frequency
	 */
	value_type operator() (value_type u) const noexcept {
		return std::pow(math::jinc_pi(xt::numeric_constants<value_type>::PI * u), 2);
	}

	/**
	 * @brief Call operator for circular aperture filter in Cartesian coordinates
	 *
	 * Evaluates the filter by converting to radial coordinates:
	 * \f$ A(u_x, u_y) = A\left(\sqrt{u_x^2 + u_y^2}\right) \f$.
	 *
	 * @param ux Dimensionless spatial frequency component in x-direction
	 * @param uy Dimensionless spatial frequency component in y-direction
	 * @return Aperture filter value at specified frequency coordinates
	 */
	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}


	/**
	 * @brief Call operator for circular aperture filter with tensor input (radial coordinates)
	 *
	 * Evaluates the squared jinc function for an array of frequency magnitudes:
	 * \f$ A(u) = \mathrm{jinc}_1^2(\pi u) \f$.
	 *
	 * @param e Input tensor of dimensionless spatial frequency magnitudes
	 * @return Tensor of aperture filter values with same shape as input
	 */
	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return xt::square(math::jinc_pi(static_cast<xvalue_type>(PI) * std::forward<E>(e)));
	}

	/**
	 * @brief Call operator for circular aperture filter with tensor inputs (Cartesian coordinates)
	 *
	 * Evaluates the filter on a grid by converting to radial coordinates:
	 * \f$ A(u_x, u_y) = A\left(\sqrt{u_x^2 + u_y^2}\right) \f$.
	 *
	 * @param ex Tensor of dimensionless x-component frequencies
	 * @param ey Tensor of dimensionless y-component frequencies
	 * @return (Nx, Ny) shaped tensor of aperture filter values
	 */
	template<class EX, class EY, xt::enable_xexpression<EX, bool> = true, xt::enable_xexpression<EY, bool> = true>
	auto operator() (EX&& ex, EY&& ey) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<EX>(ex), std::forward<EY>(ey));

		return this->operator()(xt::sqrt(xt::square(std::move(xx)) + xt::square(std::move(yy))));
	}
};

/**
 * @brief Aperture filter function for an annular (ring-shaped) aperture
 *
 * @tparam T Numeric type used for calculations.
 *
 * The aperture filter accounts for central obscuration and is defined as:
 * \f[
 * A(u) = \frac{\left(\mathrm{jinc}_1(\pi u) - \epsilon^2 \mathrm{jinc}_1(\pi \epsilon u)\right)^2}{(1 - \epsilon^2)^2},
 * \f]
 * \f[
 * A(u_x, u_y) = A\left(\sqrt{u_x^2 + u_y^2}\right).
 * \f]
 * where:
 * - \f$\epsilon\f$ is the obscuration ratio (\f$0 \le \epsilon < 1\f$),
 * - \f$\mathrm{jinc}_1(x) = \frac{2 J_1(x)}{x}\f$ is the jinc function.
 */
template<class T>
class WEIF_EXPORT annular {
public:
	using value_type = T; ///< Numeric type used for calculations

private:
	value_type obscuration_; ///< Central obscuration ratio (\f$\epsilon\f$)

public:
	/**
	 * @brief Constructs an annular aperture filter with given obscuration
	 *
	 * @param obscuration Central obscuration ratio (\f$0 \le \epsilon < 1\f$)
	 */
	explicit annular(value_type obscuration) noexcept:
		obscuration_{obscuration} {}

	/**
	 * @brief Returns the central obscuration ratio
	 *
	 * @return Current obscuration ratio (\f$\epsilon\f$)
	 */
	const value_type& obscuration() const noexcept { return obscuration_; }

	/**
	 * @brief Call operator for annular aperture in radial coordinates
	 *
	 * Evaluates the squared normalized difference of jinc functions:
	 * \f[
	 * A(u) = \frac{\left(\mathrm{jinc}_1(\pi u) - \epsilon^2 \mathrm{jinc}_1(\pi \epsilon u)\right)^2}{(1 - \epsilon^2)^2}.
	 * \f]
	 *
	 * @param u Dimensionless spatial frequency magnitude
	 * @return Aperture filter value at specified frequency
	 */
	value_type operator() (value_type u) const noexcept {
		const auto eps2 = std::pow(obscuration(), 2);
		const auto norm = std::pow(static_cast<value_type>(1) - eps2, 2);
		const auto piu = xt::numeric_constants<value_type>::PI * u;

		return std::pow(math::jinc_pi(piu) - eps2 * math::jinc_pi(obscuration() * piu), 2) / norm;
	}

	/**
	 * @brief Call operator for annular aperture in Cartesian coordinates
	 *
	 * Evaluates the filter by converting to radial coordinates:
	 * \f[
	 * A(u_x, u_y) = A\left(\sqrt{u_x^2 + u_y^2}\right).
	 * \f]
	 *
	 * @param ux Dimensionless x-component frequency
	 * @param uy Dimensionless y-component frequency
	 * @return Aperture filter value at specified coordinates
	 */
	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

	/**
	 * @brief Call operator for annular aperture with tensor input (radial coordinates)
	 *
	 * Vectorized evaluation for array of frequencies:
	 * \f[
	 * A(u) = \frac{\left(\mathrm{jinc}_1(\pi u) - \epsilon^2 \mathrm{jinc}_1(\pi \epsilon u)\right)^2}{(1 - \epsilon^2)^2}.
	 * \f]
	 *
	 * @param e Input tensor of frequency magnitudes
	 * @return Tensor of filter values with same shape as input
	 */
	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		const auto eps2 = std::pow(obscuration(), 2);
		const auto norm = std::pow(static_cast<value_type>(1) - eps2, 2);

		auto fnct = [=](auto u) -> decltype(u) {
			constexpr auto PI = xt::numeric_constants<value_type>::PI;

			return math::jinc_pi(static_cast<xvalue_type>(PI) * u) - static_cast<xvalue_type>(eps2) * math::jinc_pi(static_cast<xvalue_type>(PI * obscuration()) * u);
		};

		/* Use static_cast<> to capture by value */
		return xt::square(xt::make_lambda_xfunction(std::move(fnct), std::forward<E>(e))) / static_cast<xvalue_type>(norm);
	}

	/**
	 * @brief Call operator for annular aperture with tensor inputs (Cartesian coordinates)
	 *
	 * Evaluates the filter on a grid by converting to radial coordinates:
	 * \f[
	 * A(u_x, u_y) = A\left(\sqrt{u_x^2 + u_y^2}\right).
	 * \f]
	 *
	 * @param ex Tensor of dimensionless x-component frequencies
	 * @param ey Tensor of dimensionless y-component frequencies
	 * @return (Nx, Ny) shaped tensor of filter values
	 */
	template<class EX, class EY, xt::enable_xexpression<EX, bool> = true, xt::enable_xexpression<EY, bool> = true>
	auto operator() (EX&& ex, EY&& ey) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<EX>(ex), std::forward<EY>(ey));

		return this->operator()(xt::sqrt(xt::square(std::move(xx)) + xt::square(std::move(yy))));
	}
};


/**
 * @brief Aperture filter for covariance between two concentric annular apertures
 *
 * @tparam T Numeric type used for calculations.
 *
 * Models the covariance of relative flux fluctuations between two annular apertures
 * as used in MASS (Multi-Aperture Scintillation Sensor) instruments:
 * \f[
 * A(u) = \frac{\left(\mathrm{jinc}_1(\pi u) - \epsilon_1^2 \mathrm{jinc}_1(\pi \epsilon_1 u)\right)}{(1 - \epsilon_1^2)}
 * \times \frac{\left(\mathrm{jinc}_1(\pi \alpha u) - \epsilon_2^2 \mathrm{jinc}_1(\pi \epsilon_2 \alpha u)\right)}{(1 - \epsilon_2^2)}
 * \f]
 * where:
 * - \f$\epsilon_1\f$ is the central obscuration ratio of the first aperture,
 * - \f$\epsilon_2\f$ is the central obscuration ratio of the second aperture,
 * - \f$\alpha = D_2/D_1\f$ is the diameter ratio between apertures.
 */
template<class T>
class WEIF_EXPORT cross_annular {
public:
	using value_type = T; ///< Numeric type used for covariance calculations

private:
	value_type ratio_; ///< Diameter ratio \f$\alpha = \frac{D_2}{D_1}\f$ between apertures
	value_type obscuration_first_; ///< Obscuration ratio of first aperture (\f$\epsilon_1\f$)
	value_type obscuration_second_; ///< Obscuration ratio of second aperture (\f$\epsilon_2\f$)

	value_type calc(value_type u, value_type obscuration) const noexcept {
		const auto eps2 = std::pow(obscuration, 2);
		const auto norm = static_cast<value_type>(1) - eps2;
		const auto piu = xt::numeric_constants<value_type>::PI * u;

		return (math::jinc_pi(piu) - eps2 * math::jinc_pi(obscuration * piu)) / norm;
	}

public:
	/**
	 * @brief Constructs an aperture filter for covariance of two annular apertures
	 *
	 * @param ratio Diameter ratio \f$\alpha = \frac{D_2}{D_1}\f$ between apertures
	 * @param obscuration_first Obscuration ratio \f$\epsilon_1\f$ of first aperture (\f$0 \le \epsilon_1 < 1\f$)
	 * @param obscuration_second Obscuration ratio \f$\epsilon_2\f$ of second aperture (\f$0 \le \epsilon_2 < 1\f$)
	 */
	explicit cross_annular(value_type ratio, value_type obscuration_first, value_type obscuration_second) noexcept:
		ratio_{ratio},
		obscuration_first_{obscuration_first},
		obscuration_second_{obscuration_second} {}

	/**
	 * @brief Returns the diameter ratio
	 *
	 * @return Current diameter ratio \f$\alpha = \frac{D_2}{D_1}\f$ between apertures
	 */
	const value_type& ratio() const noexcept { return ratio_; }

	/**
	 * @brief Returns the central obscuration ratio of first aperture
	 *
	 * @return Current obscuration ratio of first aperture (\f$\epsilon_1\f$)
	 */
	const value_type& obscuration_first() const noexcept { return obscuration_first_; }

	/**
	 * @brief Returns the central obscuration ratio of second aperture
	 *
	 * @return Current obscuration ratio of second aperture (\f$\epsilon_2\f$)
	 */
	const value_type& obscuration_second() const noexcept { return obscuration_second_; }

	/**
	 * @brief Call operator for the aperture filter in radial coordinates
	 *
	 * Evaluates the filter in radial coordinates.
	 *
	 * @param u Dimensionless spatial frequency magnitude
	 * @return Aperture filter value at specified frequency
	 */
	value_type operator() (value_type u) const noexcept {
		return calc(u, obscuration_first()) * calc(u * ratio(), obscuration_second());
	}

	/**
	 * @brief Call operator for the aperture filter in Cartesian coordinates
	 *
	 * Evaluates the filter by converting to radial coordinates:
	 * \f[
	 * A(u_x, u_y) = A\left(\sqrt{u_x^2 + u_y^2}\right).
	 * \f]
	 *
	 * @param ux Dimensionless x-component frequency
	 * @param uy Dimensionless y-component frequency
	 * @return Aperture filter value at specified coordinates
	 */
	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

	/**
	 * @brief Call operator for the aperture filter with tensor input (radial coordinates)
	 *
	 * Vectorized evaluation for array of frequencies.
	 *
	 * @param e Input tensor of frequency magnitudes
	 * @return Tensor of filter values with same shape as input
	 */
	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		const auto obscuration1 = obscuration_first();
		const auto eps12 = std::pow(obscuration1, 2);
		const auto norm1 = static_cast<value_type>(1) - eps12;
		const auto obscuration2 = obscuration_second();
		const auto eps22 = std::pow(obscuration2, 2);
		const auto norm2 = static_cast<value_type>(1) - eps22;
		const auto r = ratio();

		auto fnct = [=](auto u) -> decltype(u * u) {
			constexpr auto PI = xt::numeric_constants<value_type>::PI;

			const auto a1 = math::jinc_pi(static_cast<xvalue_type>(PI) * u) - static_cast<xvalue_type>(eps12) * math::jinc_pi(static_cast<xvalue_type>(PI * obscuration1) * u);
			const auto a2 = math::jinc_pi(static_cast<xvalue_type>(PI * r) * u) - static_cast<xvalue_type>(eps22) * math::jinc_pi(static_cast<xvalue_type>(PI * obscuration2 * r) * u);

			return a1 * a2;
		};

		return xt::make_lambda_xfunction(std::move(fnct), std::forward<E>(e)) / static_cast<xvalue_type>(norm1 * norm2);
	}

	/**
	 * @brief Call operator for annular aperture with tensor inputs (Cartesian coordinates)
	 *
	 * Evaluates the filter on a grid by converting to radial coordinates:
	 * \f[
	 * A(u_x, u_y) = A\left(\sqrt{u_x^2 + u_y^2}\right).
	 * \f]
	 *
	 * @param ex Tensor of dimensionless x-component frequencies
	 * @param ey Tensor of dimensionless y-component frequencies
	 * @return (Nx, Ny) shaped tensor of filter values
	 */
	template<class EX, class EY, xt::enable_xexpression<EX, bool> = true, xt::enable_xexpression<EY, bool> = true>
	auto operator() (EX&& ex, EY&& ey) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<EX>(ex), std::forward<EY>(ey));

		return this->operator()(xt::sqrt(xt::square(std::move(xx)) + xt::square(std::move(yy))));
	}
};

} // af
} // weif

#endif // _WEIF_AF_CIRCULAR_H

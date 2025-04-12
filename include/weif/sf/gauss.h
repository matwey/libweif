/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_SF_GAUSS_H
#define _WEIF_SF_GAUSS_H

#include <cmath>
#include <type_traits>

#include <xtensor/xmath.hpp>
#include <xtensor/xutils.hpp>
#include <xtensor/xvectorize.hpp>

#include <weif/math.h>
#include <weif_export.h>


namespace weif {
namespace sf {

template<class T>
class WEIF_EXPORT gauss {
public:
	using value_type = T;

private:
	value_type fwhm_;

public:
	explicit gauss(value_type fwhm) noexcept:
		fwhm_{fwhm} {}

	const auto& fwhm() const noexcept { return fwhm_; }

	value_type operator() (const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;
		constexpr auto C = PI * PI / xt::numeric_constants<value_type>::LN2 / 8;

		return pow(sin(PI * x), 2) * exp(-C * pow(fwhm() * x, 2));
	}

	value_type regular(const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;
		using boost::math::sinc_pi;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;
		constexpr auto C = PI * PI / xt::numeric_constants<value_type>::LN2 / 8;

		return pow(PI * sinc_pi(PI * x), 2) * exp(-C * pow(fwhm() * x, 2));
	}

	template<class E, xt::enable_xexpression<std::decay_t<E>, bool> = true>
	auto operator() (E&& e) const noexcept {
		const auto fwhm_ = fwhm();

		auto fnct = [=](auto x) -> decltype(x) {
			using namespace std;

			constexpr auto PI = xt::numeric_constants<value_type>::PI;
			constexpr auto C = PI * PI / xt::numeric_constants<value_type>::LN2 / 8;

			return pow(sin(PI * x), 2) * exp(-C * pow(fwhm_ * x, 2));
		};

		return xt::make_lambda_xfunction(std::move(fnct), std::forward<E>(e));
	}

	template<class E, xt::enable_xexpression<std::decay_t<E>, bool> = true>
	auto regular(E&& e) const noexcept {
		const auto fwhm_ = fwhm();

		auto fnct = [=](auto x) -> decltype(x) {
			using namespace std;
			using boost::math::sinc_pi;

			constexpr auto PI = xt::numeric_constants<value_type>::PI;
			constexpr auto C = PI * PI / xt::numeric_constants<value_type>::LN2 / 8;

			return pow(PI * sinc_pi(PI * x), 2) * exp(-C * pow(fwhm_ * x, 2));
		};

		return xt::make_lambda_xfunction(std::move(fnct), std::forward<E>(e));
	}
};

} // sf
} // weif

#endif // _WEIF_SF_GAUSS_H


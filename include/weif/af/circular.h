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

template<class T>
struct WEIF_EXPORT circular {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		return std::pow(math::jinc_pi(xt::numeric_constants<value_type>::PI * u), 2);
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return xt::square(math::jinc_pi(static_cast<xvalue_type>(PI) * std::forward<E>(e)));
	}

	template<class E1, class E2, xt::enable_xexpression<E1, bool> = true, xt::enable_xexpression<E2, bool> = true>
	auto operator() (E1&& e1, E2&& e2) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<E1>(e1), std::forward<E2>(e2));

		return this->operator()(xt::sqrt(xt::square(std::move(xx)) + xt::square(std::move(yy))));
	}
};

template<class T>
class WEIF_EXPORT annular {
public:
	using value_type = T;

private:
	value_type obscuration_;

public:
	explicit annular(value_type obscuration) noexcept:
		obscuration_{obscuration} {}

	const auto& obscuration() const noexcept { return obscuration_; }

	value_type operator() (value_type u) const noexcept {
		const auto eps2 = std::pow(obscuration(), 2);
		const auto norm = std::pow(static_cast<value_type>(1) - eps2, 2);
		const auto piu = xt::numeric_constants<value_type>::PI * u;

		return std::pow(math::jinc_pi(piu) - eps2 * math::jinc_pi(obscuration() * piu), 2) / norm;
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

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

	template<class E1, class E2, xt::enable_xexpression<E1, bool> = true, xt::enable_xexpression<E2, bool> = true>
	auto operator() (E1&& e1, E2&& e2) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<E1>(e1), std::forward<E2>(e2));

		return this->operator()(xt::sqrt(xt::square(std::move(xx)) + xt::square(std::move(yy))));
	}
};

template<class T>
class WEIF_EXPORT cross_annular {
public:
	using value_type = T;

private:
	value_type ratio_;
	value_type obscuration_first_;
	value_type obscuration_second_;

	value_type calc(value_type u, value_type obscuration) const noexcept {
		const auto eps2 = std::pow(obscuration, 2);
		const auto norm = static_cast<value_type>(1) - eps2;
		const auto piu = xt::numeric_constants<value_type>::PI * u;

		return (math::jinc_pi(piu) - eps2 * math::jinc_pi(obscuration * piu)) / norm;
	}

public:
	explicit cross_annular(value_type ratio, value_type obscuration_first, value_type obscuration_second) noexcept:
		ratio_{ratio},
		obscuration_first_{obscuration_first},
		obscuration_second_{obscuration_second} {}

	const auto& ratio() const noexcept { return ratio_; }
	const auto& obscuration_first() const noexcept { return obscuration_first_; }
	const auto& obscuration_second() const noexcept { return obscuration_second_; }

	value_type operator() (value_type u) const noexcept {
		return calc(u, obscuration_first()) * calc(u * ratio(), obscuration_second());
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

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

	template<class E1, class E2, xt::enable_xexpression<E1, bool> = true, xt::enable_xexpression<E2, bool> = true>
	auto operator() (E1&& e1, E2&& e2) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<E1>(e1), std::forward<E2>(e2));

		return this->operator()(xt::sqrt(xt::square(std::move(xx)) + xt::square(std::move(yy))));
	}
};

} // af
} // weif

#endif // _WEIF_AF_CIRCULAR_H

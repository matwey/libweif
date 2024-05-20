/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_CIRCULAR_H
#define _WEIF_AF_CIRCULAR_H

#include <cmath>

#include <xtensor/xmath.hpp>
#include <xtensor/xvectorize.hpp>

#include <weif/af/detail/impl.h>
#include <weif_export.h>


namespace weif {
namespace af {

template<class T>
struct WEIF_EXPORT circular {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		return std::pow(detail::airy_tmp(xt::numeric_constants<value_type>::PI * u), 2);
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		const auto airy_vec = xt::vectorize(&detail::airy_tmp<value_type>);

		return xt::square(airy_vec(xt::numeric_constants<value_type>::PI * e.derived_cast()));
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E2>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
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

		return std::pow(detail::airy_tmp(piu) - eps2 * detail::airy_tmp(obscuration() * piu), 2) / norm;
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		const auto eps2 = std::pow(obscuration(), 2);
		const auto norm = std::pow(static_cast<value_type>(1) - eps2, 2);
		const auto airy_vec = xt::vectorize(&detail::airy_tmp<value_type>);

		return xt::square(airy_vec(xt::numeric_constants<value_type>::PI * e.derived_cast()) - eps2 * airy_vec((xt::numeric_constants<value_type>::PI * obscuration()) * e.derived_cast())) / norm;
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E2>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
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

		return (detail::airy_tmp(piu) - eps2 * detail::airy_tmp(obscuration * piu)) / norm;
	}

	template<class E>
	auto calc(const xt::xexpression<E>& e, value_type obscuration) const noexcept {
		const auto eps2 = std::pow(obscuration, 2);
		const auto norm = static_cast<value_type>(1) - eps2;
		const auto airy_vec = xt::vectorize(&detail::airy_tmp<value_type>);

		return (airy_vec(xt::numeric_constants<value_type>::PI * e.derived_cast()) - eps2 * airy_vec((xt::numeric_constants<value_type>::PI * obscuration) * e.derived_cast())) / norm;
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

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		auto e2 = xt::make_xshared(e.derived_cast() * ratio());

		return calc(e, obscuration_first()) * calc(e2, obscuration_second());
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E2>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

} // af
} // weif

#endif // _WEIF_AF_CIRCULAR_H

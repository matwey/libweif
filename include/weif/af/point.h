/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_POINT_H
#define _WEIF_AF_POINT_H

#include <weif_export.h>


namespace weif {
namespace af {

template<class T>
struct WEIF_EXPORT point {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		return static_cast<value_type>(1);
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return static_cast<value_type>(1);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::ones_like(e);
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E2>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

} // af
} // weif

#endif // _WEIF_AF_POINT_H

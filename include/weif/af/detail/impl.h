/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_AF_DETAIL_IMPL_H
#define _WEIF_AF_DETAIL_IMPL_H

#include <cmath>

#include <boost/math/tools/precision.hpp>


namespace weif {
namespace af {
namespace detail {

template<class T>
inline T airy_tmp(const T x) noexcept {
	using namespace std;

	if (abs(x) >= 3.7 * boost::math::tools::forth_root_epsilon<T>()) {
		return cyl_bessel_j(1, x) / x * static_cast<T>(2);
	} else {
        	// |x| < (eps*192)^(1/4)
		return static_cast<T>(1) - x * x / 8;
	}
}

} // detail
} // af
} // weif

#endif // _WEIF_AF_DETAIL_IMPL_H

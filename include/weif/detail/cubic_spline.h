/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_DETAIL_CUBIC_SPLINE_H
#define _WEIF_DETAIL_CUBIC_SPLINE_H

#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <variant>

#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>


namespace weif {
namespace detail {

template<class T>
struct first_order_boundary {
	using value_type = T;

	value_type left;
	value_type right;
};

template<class T>
first_order_boundary(T, T) -> first_order_boundary<std::decay_t<T>>;

template<class T>
struct second_order_boundary {
	using value_type = T;

	value_type left;
	value_type right;

	constexpr second_order_boundary():
		left{0},
		right{0} {}

	constexpr second_order_boundary(value_type left, value_type right):
		left{left},
		right{right} {}
};

template<class T>
second_order_boundary(T, T) -> second_order_boundary<std::decay_t<T>>;

template<class T> class cubic_spline {
public:
	using value_type = T;
	using boundary_type = std::variant<first_order_boundary<T>, second_order_boundary<T>>;

private:
	xt::xtensor<T, 1> values_;
	xt::xtensor<T, 1> d2_;

	void init_d2(const boundary_type& boundary) {
		const std::size_t n = values_.size();

		assert(n > 1);

		const auto [first, last, d0, dn] = std::visit([&] (const auto& boundary) -> std::array<value_type, 4> {
			using variant_type = std::decay_t<decltype(boundary)>;

			if constexpr (std::is_same_v<variant_type, first_order_boundary<value_type>>) {
				return {1, 1,
					(values_(1) - values_(0) - boundary.left) * 6,
					(boundary.right - (values_(n - 1) - values_(n - 2))) * 6};
			} else if (std::is_same_v<variant_type, second_order_boundary<value_type>>) {
				return {0, 0, boundary.left * 2, boundary.right * 2};
			}
		}, boundary);

		xt::xtensor<T, 1> cprime{std::array{n - 1}};

		const auto d = (
			xt::view(values_, xt::range(2, n)) -
			xt::view(values_, xt::range(1, n - 1)) * static_cast<value_type>(2) +
			xt::view(values_, xt::range(0, n - 2))) * static_cast<value_type>(3);

		/* left boundary */
		cprime(0) = first / static_cast<value_type>(2);
		d2_(0) = d0 / static_cast<value_type>(2);

		/* forward sweep */
		std::size_t i = 1;
		for (; i < n - 1; ++i) {
			const auto denom = static_cast<value_type>(2) - static_cast<value_type>(0.5) * cprime(i-1);
			cprime(i) = static_cast<value_type>(0.5) / denom;
			d2_(i) = (d(i-1) - static_cast<value_type>(0.5) * d2_(i-1)) / denom;
		}

		/* right boundary */
		d2_(i) = (dn - last * d2_(i-1)) / (static_cast<value_type>(2) - last * cprime(i-1));

		/* back substitution */
		for (; i > 0; --i) {
			d2_(i-1) = d2_(i-1) - cprime(i-1) * d2_(i);
		}
	}

public:
	template<class E>
	cubic_spline(const xt::xexpression<E>& e, const boundary_type& boundary = second_order_boundary<T>{}):
		values_{e.derived_cast()},
		d2_{values_.shape()} {

		init_d2(boundary);
	}

	const auto& values() const noexcept { return values_; }
	const auto& double_primes() const noexcept { return d2_; }

	auto size() const noexcept { return values_.size(); }

	value_type operator() (const value_type x) const noexcept {
		const auto idx = static_cast<std::size_t>(x);
		const auto delta0 = x - static_cast<value_type>(idx);
		const auto delta1 = static_cast<value_type>(1) - delta0;
		const auto delta03 = std::pow(delta0, 3);
		const auto delta13 = std::pow(delta1, 3);
		const auto d20 = d2_(idx) / 6;
		const auto d21 = d2_(idx+1) / 6;
		const auto y0 = values_(idx);
		const auto y1 = values_(idx+1);

		return d20 * delta13 + d21 * delta03 + (y0 - d20) * delta1 + (y1 - d21) * delta0;
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->operator()(x);
		}, e.derived_cast());
	}

	template<class U>
	auto operator+ (const U x) const noexcept {
		const auto ret{*this};

		ret += x;

		return ret;
	}
	template<class U>
	auto operator- (const U x) const noexcept {
		const auto ret{*this};

		ret -= x;

		return ret;
	}
	template<class U>
	auto operator* (const U x) const noexcept {
		const auto ret{*this};

		ret *= x;

		return ret;
	}
	template<class U>
	auto operator/ (const U x) const noexcept {
		const auto ret{*this};

		ret /= x;

		return ret;
	}

	template<class U>
	cubic_spline<T>& operator+= (const U x) noexcept {
		values_ += x;

		return *this;
	}
	template<class U>
	cubic_spline<T>& operator-= (const U x) noexcept {
		values_ -= x;

		return *this;
	}
	template<class U>
	cubic_spline<T>& operator*= (const U x) noexcept {
		values_ *= x;
		d2_ *= x;

		return *this;
	}
	template<class U>
	cubic_spline<T>& operator/= (const U x) noexcept {
		values_ /= x;
		d2_ /= x;

		return *this;
	}
};

template<class E>
cubic_spline(const xt::xexpression<E>& e) ->
	cubic_spline<typename std::decay_t<E>::value_type>;

template<class E>
cubic_spline(const xt::xexpression<E>& e, const typename cubic_spline<typename std::decay_t<E>::value_type>::boundary_type&) ->
	cubic_spline<typename std::decay_t<E>::value_type>;

} // detail
} // weif

#endif // _WEIF_DETAIL_CUBIC_SPLINE_H

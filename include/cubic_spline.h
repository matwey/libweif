#ifndef _WEIF_CUBIC_SPLINE_H
#define _WEIF_CUBIC_SPLINE_H

#include <cmath>
#include <type_traits>

#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>


namespace weif {

template<class T> class cubic_spline {
public:
	using value_type = T;

	struct first_order_boundary {
		value_type left;
		value_type right;
	};

	struct second_order_boundary {
		value_type left;
		value_type right;
	};

private:
	xt::xtensor<T, 1> d2_;
	xt::xtensor<T, 1> values_;

	template<class E>
	cubic_spline(E&& e, value_type first, value_type last, value_type d0, value_type dn):
		d2_{std::array{e.size()}},
		values_{std::forward<E>(e)} {

		assert(values_.size() > 1);

		xt::xtensor<T, 1> cprime{std::array{values_.size() - 1}};

		const auto d = (
			xt::view(values_, xt::range(2, values_.size())) -
			xt::view(values_, xt::range(1, values_.size() - 1)) * static_cast<value_type>(2) +
			xt::view(values_, xt::range(0, values_.size() - 2))) * static_cast<value_type>(3);

		/* left boundary */
		cprime(0) = first / static_cast<value_type>(2);
		d2_(0) = d0 / static_cast<value_type>(2);

		/* forward sweep */
		std::size_t i = 1;
		for (; i < values_.size() - 1; ++i) {
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
	cubic_spline(E&& e):
		cubic_spline(std::forward<E>(e),
			static_cast<value_type>(0),
			static_cast<value_type>(0),
			static_cast<value_type>(0),
			static_cast<value_type>(0)) {}

	template<class E>
	cubic_spline(E&& e, first_order_boundary boundary):
		cubic_spline(std::forward<E>(e),
			static_cast<value_type>(1),
			static_cast<value_type>(1),
			(e(1) - e(0) - boundary.left) * 6,
			(boundary.right - (e(e.size() - 1) - e(e.size() - 2))) * 6) {}

	template<class E>
	cubic_spline(E&& e, second_order_boundary boundary):
		cubic_spline(std::forward<E>(e),
			static_cast<value_type>(0),
			static_cast<value_type>(0),
			boundary.left * 2,
			boundary.right * 2) {}

	value_type operator() (const value_type x) const {
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

	template<class E, typename = std::enable_if_t<xt::is_xexpression<E>::value>>
	auto operator() (E&& e) const {
		return xt::make_lambda_xfunction([this] (auto x) {
			return this->operator() (x);
		}, std::forward<E>(e));
	}
};

template<class E>
cubic_spline(E&& e) -> cubic_spline<typename std::decay_t<E>::value_type>;
template<class E>
cubic_spline(E&& e, typename cubic_spline<typename std::decay_t<E>::value_type>::first_order_boundary boundary) -> cubic_spline<typename std::decay_t<E>::value_type>;
template<class E>
cubic_spline(E&& e, typename cubic_spline<typename std::decay_t<E>::value_type>::second_order_boundary boundary) -> cubic_spline<typename std::decay_t<E>::value_type>;

} // weif

#endif // _WEIF_CUBIC_SPLINE_H

#ifndef _WEIF_APERTURE_FILTER_H
#define _WEIF_APERTURE_FILTER_H

#include <cmath>
#include <type_traits>

#include <boost/math/tools/precision.hpp>

#include <xtensor/xmath.hpp>
#include <xtensor/xvectorize.hpp>


namespace weif {

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

template<class T>
struct point_aperture {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		return static_cast<value_type>(1);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::ones_like(e);
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E1>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

template<class T>
struct circular_aperture {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		return std::pow(detail::airy_tmp(xt::numeric_constants<value_type>::PI * u), 2);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		auto airy_vec = xt::vectorize(&detail::airy_tmp<value_type>);

		return xt::square(airy_vec(xt::numeric_constants<value_type>::PI * e.derived_cast()));
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E1>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

template<class T>
class annular_aperture {
public:
	using value_type = T;

private:
	value_type obscuration_;

public:
	explicit annular_aperture(value_type obscuration) noexcept:
		obscuration_{obscuration} {}

	const auto& obscuration() const noexcept { return obscuration_; }

	value_type operator() (value_type u) const noexcept {
		const auto eps2 = std::pow(obscuration(), 2);
		const auto norm = std::pow(static_cast<value_type>(1) - eps2, 2);
		const auto piu = xt::numeric_constants<value_type>::PI * u;

		return std::pow(detail::airy_tmp(piu) - eps2 * detail::airy_tmp(obscuration() * piu), 2) / norm;
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		const auto eps2 = std::pow(obscuration(), 2);
		const auto norm = std::pow(static_cast<value_type>(1) - eps2, 2);

		auto airy_vec = xt::vectorize(&detail::airy_tmp<value_type>);

		return xt::square(airy_vec(xt::numeric_constants<value_type>::PI * e.derived_cast()) - eps2 * airy_vec((xt::numeric_constants<value_type>::PI * obscuration()) * e.derived_cast())) / norm;
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E1>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

} // weif

#endif // _WEIF_APERTURE_FILTER_H

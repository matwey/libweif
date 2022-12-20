#ifndef _WEIF_APERTURE_FILTER_H
#define _WEIF_APERTURE_FILTER_H

#include <cmath>
#include <memory>
#include <type_traits>

#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/math/tools/precision.hpp>

#include <xtensor/xmath.hpp>
#include <xtensor/xvectorize.hpp>

#include <cubic_spline.h>
#include <uniform_grid.h>

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

template<class T>
struct point {
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
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E1>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

template<class T>
struct circular {
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
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E1>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

template<class T>
class annular {
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
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E1>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

template<class T>
struct square {
	using value_type = T;

	value_type operator() (value_type ux, value_type uy) const noexcept {
		using namespace std;
		using boost::math::sinc_pi;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(sinc_pi(ux * PI) * sinc_pi(uy * PI), 2);
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E1>& e2) const noexcept {
		using boost::math::sinc_pi;

		const auto sinc_pi_vec = xt::vectorize(&sinc_pi<value_type>);

		return xt::square(sinc_pi_vec(xt::numeric_constants<value_type>::PI * e1.derived_cast()) * sinc_pi_vec(xt::numeric_constants<value_type>::PI * xt::expand_dims(e2.derived_cast(), 1)));
	}
};

template<class T>
class angle_averaged {
public:
	using value_type = T;

private:
	uniform_grid<value_type> grid_;
	cubic_spline<value_type> af_;

	template<class AF>
	static auto integrate_aperture_function(AF&& aperture_filter, std::size_t size) {
		using boost::math::quadrature::tanh_sinh;

		auto integrator = std::make_unique<tanh_sinh<value_type>>();

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return xt::make_lambda_xfunction([integrator = std::move(integrator), aperture_filter = std::forward<AF>(aperture_filter)] (value_type z) -> value_type {
			if (z == static_cast<value_type>(0))
				return static_cast<value_type>(0);

			const auto tol = std::pow(std::numeric_limits<value_type>::epsilon(), static_cast<value_type>(2.0/3.0));
			const auto u = (static_cast<value_type>(1) - z) / z;

			return integrator->integrate([u, &aperture_filter] (value_type t) noexcept {
				using namespace std;

				const auto f = PI * (t + static_cast<value_type>(1));
				const auto ux = u * cos(f);
				const auto uy = u * sin(f);

				return aperture_filter(ux, uy);
			}, tol) / 2;
		}, xt::linspace(static_cast<value_type>(0), static_cast<value_type>(1), size));
	}

	template<class E>
	explicit angle_averaged(const xt::xexpression<E>& values):
		grid_{static_cast<value_type>(0), static_cast<value_type>(1) / (values.derived_cast().size() - 1), values.derived_cast().size()},
		af_{values, typename cubic_spline<T>::first_order_boundary{0, 0}} {}

public:
	template<class AF>
	angle_averaged(AF&& aperture_filter, std::size_t size):
		angle_averaged(integrate_aperture_function(std::forward<AF>(aperture_filter), size)) {}

	value_type operator() (value_type u) const noexcept {
		const value_type z = (static_cast<value_type>(1) / (static_cast<value_type>(1) + u) - grid_.origin()) / grid_.delta();

		return af_(z);
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) { return this->operator()(x); }, e.derived_cast());
	}

	template<class E1, class E2>
	auto operator() (const xt::xexpression<E1>& e1, const xt::xexpression<E2>& e2) const noexcept {
		return this->operator()(xt::sqrt(xt::square(e1.derived_cast()) + xt::expand_dims(xt::square(e2.derived_cast()),1)));
	}
};

} // af
} // weif

#endif // _WEIF_APERTURE_FILTER_H

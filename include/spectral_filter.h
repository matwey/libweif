#ifndef _WEIF_SPECTRAL_FILTER_H
#define _WEIF_SPECTRAL_FILTER_H

#include <complex>
#include <string>

#include <xtensor/xcomplex.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>

#include <cubic_spline.h>
#include <spectral_response.h>
#include <uniform_grid.h>

#include <weif_export.h>


namespace weif {

template<class T>
class WEIF_EXPORT spectral_filter {
public:
	static_assert(std::is_floating_point<T>::value, "type T is not supported");

	using value_type = T;
private:
	/* Consider using different type for uniform_grid, for instance, boost::cpp_dec_float */
	uniform_grid<value_type> grid_;
	cubic_spline<value_type> real_;
	cubic_spline<value_type> imag_;
	value_type carrier_;

	template<class E>
	xt::xtensor<std::complex<value_type>, 1> make_fft(const xt::xexpression<E>& e);

	template<class E1, class E2>
	spectral_filter(const uniform_grid<value_type>& grid, const xt::xexpression<E1>& real, const xt::xexpression<E2>& imag, value_type carrier);
	template<class E>
	spectral_filter(value_type delta, const xt::xexpression<E>& e, value_type carrier);
	spectral_filter(const spectral_response<value_type>& response, std::size_t size, std::size_t carrier_idx, std::size_t padded_size);
public:
	spectral_filter(const spectral_response<value_type>& response, std::size_t size, value_type carrier);
	spectral_filter(const spectral_response<value_type>& response, std::size_t size);

	const auto& grid() const noexcept { return grid_; }
	const auto& real() const noexcept { return real_; }
	const auto& imag() const noexcept { return imag_; }
	const auto& carrier() const noexcept { return carrier_; }

	value_type operator() (const value_type x) const noexcept {
		using namespace std;

		const auto ax = abs(x);

		if (grid() <= ax)
			return static_cast<value_type>(0);

		const auto c = static_cast<value_type>(2) * xt::numeric_constants<value_type>::PI * carrier();
		const auto cx = ax * c;
		const auto dx = (ax - grid().origin()) / grid().delta();

		return pow(cos(cx) * imag()(dx) + sin(cx) * real()(dx), 2);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->operator()(x);
		}, e.derived_cast());
	}

	auto values() const noexcept {
		return operator() (grid().values());
	}

	void normalize() noexcept;
	spectral_filter<value_type> normalized() const;

	value_type equiv_lambda() const noexcept;

	void dump(const std::string& filename) const;
};

template<class T>
spectral_filter(const spectral_response<T>& response) -> spectral_filter<T>;
template<class T>
spectral_filter(const uniform_grid<T>& grid, const xt::xtensor<T, 1>& data) -> spectral_filter<T>;

} // weif

#endif // _WEIF_SPECTRAL_FILTER_H

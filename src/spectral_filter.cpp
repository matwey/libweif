#include <array>
#include <cassert>
#include <complex>
#include <fstream>
#include <iomanip>

#include <fftw3.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <spectral_filter.h>


namespace weif {

template<class T>
template<class E>
xt::xtensor<std::complex<typename spectral_filter<T>::value_type>, 1> spectral_filter<T>::make_fft(const xt::xexpression<E>& e) {
	const auto size = e.derived_cast().size();

	xt::xtensor<std::complex<value_type>, 1> ret{std::array{size / 2 + 1}};
	auto in_adapter = xt::adapt(reinterpret_cast<value_type*>(ret.data()), size, xt::no_ownership(), std::array{size});
	in_adapter.assign(e);

	if constexpr (std::is_same<value_type, float>::value) {
		fftwf_plan p = fftwf_plan_dft_r2c_1d(size,
			in_adapter.data(),
			reinterpret_cast<fftwf_complex*>(ret.data()),
			FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

		assert(p != nullptr);

		fftwf_execute(p);
		fftwf_destroy_plan(p);
	} else if (std::is_same<value_type, double>::value) {
		fftw_plan p = fftw_plan_dft_r2c_1d(size,
			in_adapter.data(),
			reinterpret_cast<fftw_complex*>(ret.data()),
			FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

		assert(p != nullptr);

		fftw_execute(p);
		fftw_destroy_plan(p);
	} else {
		static_assert(true, "value_type is not supported");
	}

	/* Boundary condition at +inf */
	ret(ret.size()-1) = std::complex<value_type>{0, 0};

	return ret;
}

template<class T>
template<class E1, class E2>
spectral_filter<T>::spectral_filter(const uniform_grid<value_type>& grid, const xt::xexpression<E1>& real, const xt::xexpression<E2>& imag, value_type carrier):
	grid_{grid},
	real_{real.derived_cast(), typename cubic_spline<T>::first_order_boundary{0, 0}},
	imag_{imag.derived_cast(), typename cubic_spline<T>::first_order_boundary{0, 0}},
	carrier_{carrier} {}

template<class T>
template<class E>
spectral_filter<T>::spectral_filter(value_type delta, const xt::xexpression<E>& e, T carrier):
	spectral_filter(uniform_grid{static_cast<value_type>(0), delta, e.derived_cast().size()},
		xt::real(e.derived_cast()),
		xt::imag(e.derived_cast()),
		carrier) {}

template<class T>
spectral_filter<T>::spectral_filter(const spectral_response<value_type>& response, std::size_t size, std::size_t carrier_idx, std::size_t padded_size):
	spectral_filter(static_cast<value_type>(1) / response.grid().delta() / padded_size,
		make_fft(xt::view(
			xt::tile(xt::pad(response.data() / response.grid().values(),
				std::vector<std::size_t>{{0, (padded_size > response.grid().size() ? padded_size - response.grid().size() : 0)}}), {2}),
			xt::range(carrier_idx, carrier_idx + padded_size))),
		response.grid().values()[carrier_idx]) {}

template<class T>
spectral_filter<T>::spectral_filter(const spectral_response<value_type>& response, std::size_t size, value_type carrier):
	spectral_filter(response, size, response.grid().to_index(carrier), std::max(response.grid().size(), size)) {}

template<class T>
spectral_filter<T>::spectral_filter(const spectral_response<value_type>& response, std::size_t size):
	spectral_filter(response, size, response.effective_lambda()) {}

template<class T>
void spectral_filter<T>::normalize() noexcept {
	const auto lambda_0 = equiv_lambda();

	grid_ *= lambda_0;
	carrier_ /= lambda_0;
	real_ *= lambda_0;
	imag_ *= lambda_0;
}

template<class T>
spectral_filter<T> spectral_filter<T>::normalized() const {
	spectral_filter<T> ret{*this};

	ret.normalize();

	return ret;
}

template<class T>
typename spectral_filter<T>::value_type spectral_filter<T>::equiv_lambda() const noexcept {
	using namespace xt::placeholders;
	using std::pow;

	/* S(0) * 0**(-11/6) == 0 and S(inf) * inf**(-11/6) == 0 */
	const auto vg = xt::view(grid_.values(), xt::range(1, _));
	const auto vs = xt::view(values(), xt::range(1, _));
	const auto i = grid_.delta() * xt::sum(vs * xt::pow(vg, -static_cast<value_type>(11.0/6.0)))();

	/* 5.38 == 3.28 * 2**(5/7) */
	return static_cast<value_type>(5.38) * pow(i, -static_cast<value_type>(6.0/7.0));
}

template<class T>
void spectral_filter<T>::dump(const std::string& filename) const {
	std::ofstream fstm{filename};
	fstm << "# 1/nm   value" << std::endl;
	fstm << std::setprecision(7);

	const auto v_expr = values();
	for (std::size_t i = 0; i < grid_.size(); ++i) {
		fstm << grid_.values()[i] << " " << v_expr[i] << std::endl;
	}
}

template class spectral_filter<float>;
template class spectral_filter<double>;

} // weif

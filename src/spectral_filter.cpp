#include <cassert>
#include <complex>
#include <fstream>
#include <iomanip>

#include <fftw3.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <spectral_filter.h>


namespace weif {

template<class T>
spectral_filter<T>::spectral_filter(const spectral_response<value_type>& response, std::size_t size):
	grid_(static_cast<value_type>(0),
		static_cast<value_type>(1) / response.grid().delta() / std::max(size, response.grid().size()),
		std::max(size, response.grid().size()) / 2 + 1) {

	const std::array<std::size_t, 1> out_shape{{grid_.size()}};
	xt::xtensor<std::complex<value_type>, 1> storage(out_shape);

	const std::array<std::size_t, 1> in_shape{{std::max(size, response.grid().size())}};
	auto in = xt::adapt(reinterpret_cast<value_type*>(storage.data()), in_shape[0], xt::no_ownership(), in_shape);
	const std::vector<std::size_t> pad{{0, in.size() - response.data().size()}};
	in = xt::pad(response.data() / response.grid().values(), pad);

	fftwf_plan p = fftwf_plan_dft_r2c_1d(size,
		in.data(),
		reinterpret_cast<fftwf_complex*>(storage.data()),
		FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

	assert(p != nullptr);

	fftwf_execute(p);
	fftwf_destroy_plan(p);

	const auto c = static_cast<value_type>(2) * xt::numeric_constants<value_type>::PI * response.grid().origin();
	data_ = xt::square(xt::cos(grid_.values() * c) * xt::imag(storage) - xt::sin(grid_.values() * c) * xt::real(storage));

	assert(data_.size() == grid_.size());
}

template<class T>
typename spectral_filter<T>::value_type spectral_filter<T>::equiv_lambda() const noexcept {
	using namespace xt::placeholders;
	using std::pow;

	/* S(0) * 0**(-11/6) == 0 and S(inf) * inf**(-11/6) == 0 */
	const auto vg = xt::view(grid_.values(), xt::range(1, _));
	const auto vs = xt::view(data_, xt::range(1, _));
	const auto i = grid_.delta() * xt::sum(vs * xt::pow(vg, -static_cast<value_type>(11.0/6.0)))();

	/* 5.38 == 3.28 * 2**(5/7) */
	return static_cast<value_type>(5.38) * pow(i, -static_cast<value_type>(6.0/7.0));
}

template<class T>
void spectral_filter<T>::dump(const std::string& filename) const {
	std::ofstream fstm{filename};
	fstm << "# 1/nm   value" << std::endl;
	fstm << std::setprecision(7);

	for (std::size_t i = 0; i < grid_.size(); ++i) {
		fstm << grid_.values()[i] << " " << data_[i] << std::endl;
	}
}

template class spectral_filter<float>;

} // weif

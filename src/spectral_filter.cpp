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
spectral_filter<T>::spectral_filter(const spectral_response<value_type>& response, std::size_t size):
	grid_(static_cast<value_type>(0),
		static_cast<value_type>(1) / response.grid().delta() / std::max(size, response.grid().size()),
		std::max(size, response.grid().size()) / 2 + 1),
	data_(std::array{grid_.size()}) {

	const std::size_t carrier_idx = response.grid().to_index(response.effective_lambda());
	carrier_ = static_cast<value_type>(2) * xt::numeric_constants<value_type>::PI * response.grid().values()[carrier_idx];

	const std::size_t in_size = std::max(size, response.grid().size());
	auto in_storage = xt::adapt(reinterpret_cast<value_type*>(data_.data()), in_size, xt::no_ownership(), std::array{in_size});

	const std::vector<std::size_t> pad{{0, in_storage.size() - response.data().size()}};
	in_storage.assign(xt::view(xt::tile(xt::pad(response.data() / response.grid().values(), pad), {2}), xt::range(carrier_idx, carrier_idx + in_size)));

	fftwf_plan p = fftwf_plan_dft_r2c_1d(size,
		in_storage.data(),
		reinterpret_cast<fftwf_complex*>(data_.data()),
		FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

	assert(p != nullptr);

	fftwf_execute(p);
	fftwf_destroy_plan(p);

	assert(data_.size() == grid_.size());
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

} // weif

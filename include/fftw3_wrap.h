#ifndef _FFTW3_WRAP_H
#define _FFTW3_WRAP_H

#include <array>
#include <cassert>
#include <complex>
#include <memory>
#include <type_traits>

#include <fftw3.h>


namespace weif {

namespace detail {

template<class T>
struct fftw_traits;

template<>
struct fftw_traits<float> {
	using value_type = float;
	using complex_type = fftwf_complex;
	using plan_type = fftwf_plan;
	constexpr static auto destroy_plan = &fftwf_destroy_plan;

	constexpr static auto plan_dft_r2c = &fftwf_plan_dft_r2c;
	constexpr static auto execute_dft_r2c = &fftwf_execute_dft_r2c;

	constexpr static auto plan_r2r = &fftwf_plan_r2r;
	constexpr static auto execute_r2r = &fftwf_execute_r2r;
};

template<>
struct fftw_traits<double> {
	using value_type = double;
	using complex_type = fftw_complex;
	using plan_type = fftw_plan;
	constexpr static auto destroy_plan = &fftw_destroy_plan;

	constexpr static auto plan_dft_r2c = &fftw_plan_dft_r2c;
	constexpr static auto execute_dft_r2c = &fftw_execute_dft_r2c;

	constexpr static auto plan_r2r = &fftw_plan_r2r;
	constexpr static auto execute_r2r = &fftw_execute_r2r;
};

template<>
struct fftw_traits<long double> {
	using value_type = long double;
	using complex_type = fftwl_complex;
	using plan_type = fftwl_plan;
	constexpr static auto destroy_plan = &fftwl_destroy_plan;

	constexpr static auto plan_dft_r2c = &fftwl_plan_dft_r2c;
	constexpr static auto execute_dft_r2c = &fftwl_execute_dft_r2c;

	constexpr static auto plan_r2r = &fftwl_plan_r2r;
	constexpr static auto execute_r2r = &fftwl_execute_r2r;
};

template<class T>
class fft_plan {
public:
	static_assert(std::is_floating_point<T>::value, "type T is not supported");
	using traits_type = fftw_traits<T>;
	using plan_type = typename traits_type::plan_type;

private:
	struct deleter {
		constexpr void operator() (plan_type plan) const noexcept {
			traits_type::destroy_plan(plan);
		}
	};

public:
	explicit fft_plan(plan_type plan) noexcept:
		plan_{plan} {

		assert(plan != nullptr);
	}

	operator plan_type() const noexcept {
		return plan_.get();
	}

protected:
	~fft_plan() = default;

private:
	std::unique_ptr<std::remove_pointer_t<plan_type>, deleter> plan_;
};

} // detail;

template<class T>
struct fft_plan_r2c:
	public detail::fft_plan<T> {
	using traits_type = detail::fftw_traits<T>;
	using value_type = T;
	using complex_type = std::complex<T>;

	template<std::size_t Rank>
	fft_plan_r2c(const std::array<int, Rank>& n, value_type* in, complex_type* out, unsigned flags) noexcept:
		detail::fft_plan<T>(traits_type::plan_dft_r2c(n.size(), n.data(),
			in, reinterpret_cast<typename traits_type::complex_type*>(out), flags)) {}

	void operator() (value_type* in, complex_type* out) const noexcept {
		traits_type::execute_dft_r2c(*this, in, reinterpret_cast<typename traits_type::complex_type*>(out));
	}
};

template<class T>
struct fft_plan_r2r:
	public detail::fft_plan<T> {
	using traits_type = detail::fftw_traits<T>;
	using value_type = T;

	template<std::size_t Rank>
	fft_plan_r2r(const std::array<int, Rank>& n, value_type* in, value_type* out, const std::array<fftw_r2r_kind, Rank>& kind, unsigned flags) noexcept:
		detail::fft_plan<T>(traits_type::plan_r2r(n.size(), n.data(), in, out, kind.data(), flags)) {}

	void operator() (value_type* in, value_type* out) const noexcept {
		traits_type::execute_r2r(*this, in, out);
	}
};

template<class T, std::size_t Rank>
fft_plan_r2c(const std::array<int, Rank>& n, T* in, std::complex<T>* out, unsigned flags) -> fft_plan_r2c<T>;

template<class T, std::size_t Rank>
fft_plan_r2r(const std::array<int, Rank>& n, T* in, T* out, const std::array<fftw_r2r_kind, Rank>& kind, unsigned flags) -> fft_plan_r2r<T>;

} // weif

#endif // _FFTW3_WRAP_H

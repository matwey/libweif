/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <variant>
#include <vector>
#include <optional>

#include <boost/program_options.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xmanipulation.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <weif/af/circular.h>
#include <weif/detail/weight_function_base.h>

#include <weif/detail/cubic_spline.h>
#include <weif/detail/fftw3_wrap.h> // IWYU pragma: keep
#include <weif/uniform_grid.h>
#include <weif/spectral_response.h>


using value_type = long double;

namespace sf {

template<class T>
class mono {
public:
	using value_type = T;

	value_type operator() (const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return pow(cos(PI * x), 2);
	}

	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		return xt::make_lambda_xfunction([this] (auto x) -> decltype(x) {
			return this->operator()(x);
		}, std::forward<E>(e));
	}
};

template<class T>
class gauss {
public:
	using value_type = T;

private:
	value_type fwhm_;

public:
	explicit gauss(value_type fwhm) noexcept:
		fwhm_{fwhm} {}

	const auto& fwhm() const noexcept { return fwhm_; }

	value_type operator() (const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;
		constexpr auto C = PI * PI / xt::numeric_constants<value_type>::LN2 / 8;

		return pow(cos(PI * x), 2) * exp(-C * pow(fwhm() * x, 2));
	}

	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		return xt::make_lambda_xfunction([this] (auto x) -> decltype(x) {
			return this->operator()(x);
		}, std::forward<E>(e));
	}
};

template<class T>
class poly {
public:
	using value_type = T;

private:
	/* Consider using different type for uniform_grid, for instance, boost::cpp_dec_float */
	weif::uniform_grid<value_type> grid_;
	weif::detail::cubic_spline<value_type> real_;
	weif::detail::cubic_spline<value_type> imag_;
	value_type carrier_;
	value_type effective_lambda_;

	template<class E>
	xt::xtensor<std::complex<value_type>, 1> make_fft(const xt::xexpression<E>& e);

	template<class E1, class E2>
	poly(const weif::uniform_grid<value_type>& grid, const xt::xexpression<E1>& real, const xt::xexpression<E2>& imag, value_type effective_lambda, value_type carrier):
		grid_{grid},
		real_{real.derived_cast(), typename weif::detail::first_order_boundary<value_type>{0, 0}},
		imag_{imag.derived_cast(), typename weif::detail::second_order_boundary<value_type>{0, 0}},
		carrier_{carrier},
		effective_lambda_{effective_lambda} {}

	template<class E>
	poly(value_type delta, const xt::xexpression<E>& e, value_type effective_lambda, value_type carrier):
		poly({static_cast<value_type>(0), delta, e.derived_cast().size()},
			xt::real(e.derived_cast()),
			xt::imag(e.derived_cast()),
			effective_lambda,
			carrier) {}

	poly(const weif::spectral_response<value_type>& response, std::size_t padded_size, value_type carrier, std::size_t carrier_idx):
		poly(static_cast<value_type>(1) / response.grid().delta() / padded_size,
			make_fft(xt::view(
				xt::tile(xt::pad(response.data(),
					std::vector<std::size_t>{{0, (padded_size > response.grid().size() ? padded_size - response.grid().size() : 0)}}), {2}),
				xt::range(carrier_idx, carrier_idx + padded_size))),
			carrier,
			response.grid().values()[carrier_idx]) {}

public:
	poly(const weif::spectral_response<value_type>& response, std::size_t size, value_type carrier):
		poly(response, std::max(response.grid().size(), size), carrier, response.grid().to_index(carrier)) {}
	poly(const weif::spectral_response<value_type>& response, std::size_t size):
		poly(response, size, response.effective_lambda()) {}

	value_type operator() (const value_type x) const noexcept {
		// x = u^2 / lambda = z f^2
		using namespace std;

		const auto ax = abs(x);

		if (grid() <= ax)
			return static_cast<value_type>(0);

		constexpr auto PI = xt::numeric_constants<value_type>::PI;
		const auto c = PI * carrier();
		const auto cx = ax * c;
		const auto dx = (ax / static_cast<value_type>(2) - grid().origin()) / grid().delta();

		return pow(cos(cx) * real()(dx) + sin(cx) * imag()(dx), 2);
	}

	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		return xt::make_lambda_xfunction([this] (auto x) -> decltype(x) {
			return this->operator()(x);
		}, std::forward<E>(e));
	}

	const auto& grid() const noexcept { return grid_; }
	const auto& real() const noexcept { return real_; }
	const auto& imag() const noexcept { return imag_; }
	const auto& carrier() const noexcept { return carrier_; }
	const auto& effective_lambda() const noexcept { return effective_lambda_; }

	poly<value_type>& normalize() noexcept;
	poly<value_type> normalized() const;
};

template<class T>
template<class E>
xt::xtensor<std::complex<typename poly<T>::value_type>, 1> poly<T>::make_fft(const xt::xexpression<E>& e) {
	const auto size = e.derived_cast().size();

	xt::xtensor<std::complex<value_type>, 1> ret{std::array{size / 2 + 1}};
	auto in_adapter = xt::adapt(reinterpret_cast<value_type*>(ret.data()), size, xt::no_ownership(), std::array{size});
	in_adapter.assign(e);

	weif::detail::fft_plan_r2c plan{std::array{static_cast<int>(size)}, in_adapter.data(), ret.data(), FFTW_ESTIMATE | FFTW_DESTROY_INPUT};
	plan(in_adapter.data(), ret.data());

	/* Boundary condition at +inf */
	ret(ret.size()-1) = std::complex<value_type>{0, 0};

	return ret;
}

template<class T>
poly<T>& poly<T>::normalize() noexcept {
	const auto lambda_0 = effective_lambda();

	grid_ *= lambda_0;
	carrier_ /= lambda_0;
	effective_lambda_ /= lambda_0;

	return *this;
}

template<class T>
poly<T> poly<T>::normalized() const {
	poly<T> ret{*this};

	ret.normalize();

	return ret;
}


template<class T>
poly(const weif::spectral_response<T>& response, std::size_t size, T carrier) -> poly<T>;
template<class T>
poly(const weif::spectral_response<T>& response, std::size_t size) -> poly<T>;
}

template<class T>
struct longitudinal {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		using boost::math::cyl_bessel_j;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return static_cast<value_type>(1) - cyl_bessel_j(0, 2 * PI * u) + cyl_bessel_j(2, 2 * PI * u);
	}
};

template<class T>
struct transversal {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		using boost::math::cyl_bessel_j;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return static_cast<value_type>(1) - cyl_bessel_j(0, 2 * PI * u) - cyl_bessel_j(2, 2 * PI * u);
	}
};

template<class T>
struct ztilt {
	using value_type = T;

	value_type operator() (value_type u) const noexcept {
		return std::pow(weif::math::zinc_pi(xt::numeric_constants<value_type>::PI * u), 2);
	}

	value_type operator() (value_type ux, value_type uy) const noexcept {
		return this->operator()(std::hypot(ux, uy));
	}

	template<class E, xt::enable_xexpression<E, bool> = true>
	auto operator() (E&& e) const noexcept {
		using xvalue_type = xt::get_value_type_t<std::decay_t<E>>;

		constexpr auto PI = xt::numeric_constants<value_type>::PI;

		return xt::square(weif::math::zinc_pi(static_cast<xvalue_type>(PI) * std::forward<E>(e)));
	}

	template<class E1, class E2, xt::enable_xexpression<E1, bool> = true, xt::enable_xexpression<E2, bool> = true>
	auto operator() (E1&& e1, E2&& e2) const noexcept {
		auto [xx, yy] = xt::meshgrid(std::forward<E1>(e1), std::forward<E2>(e2));

		return this->operator()(xt::sqrt(xt::square(std::move(xx)) + xt::square(std::move(yy))));
	}
};

template<class T>
class weight_function:
	public weif::detail::weight_function_base<T> {
public:
	using value_type = typename weif::detail::weight_function_base<T>::value_type;

private:
	using function_type = std::function<value_type(value_type, value_type, value_type)>;

private:
	static auto integrate_weight_function(function_type&& fun, std::size_t size);

	weight_function(value_type lambda, value_type aperture_scale, function_type&& fun, std::size_t size):
		weif::detail::weight_function_base<T>(lambda, aperture_scale, integrate_weight_function(std::forward<function_type>(fun), size)) {}

public:
	template<class SF, class AF, class CF>
	weight_function(SF&& spectral_filter, value_type lambda, AF&& aperture_filter, value_type aperture_scale, CF&& component_filter, value_type base_length, std::size_t size):
		weight_function(lambda, aperture_scale,
			[spectral_filter = std::forward<SF>(spectral_filter),
			aperture_filter = std::forward<AF>(aperture_filter),
			component_filter = std::forward<CF>(component_filter),
			b = base_length / aperture_scale] (value_type u, value_type x_far, value_type x_near) noexcept -> value_type {
			using namespace std;

			if (u == static_cast<value_type>(0) || u == std::numeric_limits<value_type>::infinity())
				return static_cast<value_type>(0);

			const auto q = x_far * u;
			const auto bq = b * q;

			const auto t = pow(u, -static_cast<value_type>(2.0/3.0));
			if (t == static_cast<value_type>(0))
				return static_cast<value_type>(0);

			const auto sf = spectral_filter(u * u / (x_near * x_near));

			const auto cf = component_filter(bq);
			if (cf == static_cast<value_type>(0))
				return static_cast<value_type>(0);

			const auto af = aperture_filter(q);
			if (af == static_cast<value_type>(0))
				return static_cast<value_type>(0);

			return std::cbrt(x_far) * t * sf * af * cf;
		}, size) {}

	inline value_type operator() (value_type altitude) const noexcept {
		using namespace std;

		constexpr const auto PI = xt::numeric_constants<value_type>::PI;
		/* 10 = pow(1e3, -1.0/6.0) * pow(1e9, 1.0/6.0) */
		constexpr const value_type c = 9.69e-3 * 8 * PI * PI * PI * 10;

		const value_type fresnel_radius = sqrt(this->lambda() * altitude);
		const value_type z = (fresnel_radius / (fresnel_radius + this->aperture_scale()) - this->grid().origin()) / this->grid().delta();

		return c / std::cbrt(this->aperture_scale()) * this->wf()(z);
	}

	template<class E>
	auto operator() (const xt::xexpression<E>& e) const noexcept {
		return xt::make_lambda_xfunction([this] (const auto& x) {
			return this->operator()(x);
		}, e.derived_cast());
	}
};

template<class T>
auto weight_function<T>::integrate_weight_function(function_type&& fun, std::size_t size) {
	using boost::math::quadrature::exp_sinh;

	auto integrator = std::make_unique<exp_sinh<value_type>>();

	return xt::make_lambda_xfunction([integrator = std::move(integrator), fun = std::forward<function_type>(fun)] (value_type z) -> value_type {
		using namespace std::placeholders;

		const auto tol = std::pow(std::numeric_limits<value_type>::epsilon(), static_cast<value_type>(2.0/3.0));
		const auto x = (static_cast<value_type>(1) - z) / z;
		constexpr value_type one = 1;
		const auto [x_far, x_near] = std::minmax(x, one);

		return integrator->integrate(std::bind(std::cref(fun), _1, x_far, x_near), tol);
	}, xt::linspace(static_cast<value_type>(0), static_cast<value_type>(1), size));
}

std::pair<value_type, std::variant<sf::mono<value_type>, sf::gauss<value_type>, sf::poly<value_type>>>
make_spectral_filter(const std::vector<std::string>& response_filename, std::optional<value_type> mono) {
	if (mono) {
		return {*mono, sf::mono<value_type>{}};
	}

	auto sr = weif::spectral_response<value_type>::stack_from_files(response_filename.cbegin(), response_filename.cend());
	std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
	sr.normalize();

	sf::poly sf{sr, 4096};
	const auto lambda = sf.effective_lambda();
	std::cerr << "Effective lambda: " << lambda << std::endl;
	std::cerr << "Carrier: " << sf.carrier() << std::endl;
	sf.normalize();
	std::cerr << "Effective lambda: " << sf.effective_lambda() << std::endl;
	std::cerr << "Carrier: " << sf.carrier() << std::endl;

	return {lambda, std::move(sf)};
}

int main(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size")
		("aperture_scale", po::value<value_type>()->default_value(20.574), "Aperture scale, mm.")
		("base_length", po::value<value_type>()->default_value(20.574), "Base length, mm.")
		("output_filename", po::value<std::string>()->default_value("wf.dat"), "Output filename")
		("response_filename", po::value<std::vector<std::string>>()->required(), "Spectral response input filename")
		("mono", po::value<value_type>(), "Use monochromatic spectral filter with given labmda");

	try {
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);

		if (va.count("help")) {
			std::cerr << opts << std::endl;

			return 1;
		}

		po::notify(va);

		const auto size = va["size"].as<std::size_t>();
		const auto aperture_scale = va["aperture_scale"].as<value_type>();
		const auto base_length = va["base_length"].as<value_type>();
		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();
		const auto output_filename = va["output_filename"].as<std::string>();

		const auto [lambda, spectral_filter] = make_spectral_filter(
			response_filename,
			(va.count("mono") ? std::optional{va["mono"].as<value_type>()} : std::nullopt));

		const xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(60), size);

		const auto t1 = std::chrono::high_resolution_clock::now();

		constexpr auto wf_grid_size = 1024 + 1;
		const auto af = ztilt<value_type>{};
		const auto cf = longitudinal<value_type>{};
		const auto wf = std::visit([&] (const auto& sf) {
			return weight_function<value_type>{sf, lambda, af, aperture_scale, cf, base_length, wf_grid_size};
		}, spectral_filter);

		const auto t2 = std::chrono::high_resolution_clock::now();

		std::ofstream stm(output_filename);
		xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid, wf(grid)))));

		std::cerr << "Consumed time: " << std::chrono::duration_cast<std::chrono::duration<value_type>>(t2-t1).count() << " sec" << std::endl;

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

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

#include <boost/program_options.hpp>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/io/xcsv.hpp>
#include <xtensor/misc/xmanipulation.hpp>

#include <weif/af/angle_averaged.h>
#include <weif/af/circular.h>
#include <weif/af/square.h>
#include <weif/af/point.h>
#include <weif/sf/mono.h>
#include <weif/sf/poly.h>
#include <weif/spectral_response.h>
#include <weif/weight_function.h>



template<class AF>
class dimm_aperture_function {
public:
	using inner_aperture_function = AF;
	using value_type = typename inner_aperture_function::value_type;

private:
	inner_aperture_function aperture_function_;
	value_type base_ratio_;

public:
	dimm_aperture_function(AF&& aperture_function, value_type base_ratio):
		aperture_function_(std::move(aperture_function)),
		base_ratio_(base_ratio) {}

	value_type operator() (value_type u) const noexcept {
		using boost::math::cyl_bessel_j;

		constexpr auto TWO_PI = xt::numeric_constants<value_type>::PI * 2;
		const auto af = aperture_function_(u);

		if (af == static_cast<value_type>(0)) {
			return static_cast<value_type>(0);
		}

		return af * cyl_bessel_j(0, TWO_PI * u * base_ratio_);
	}
};

template<class AF>
dimm_aperture_function(AF&&, typename AF::value_type) -> dimm_aperture_function<AF>;


using value_type = float;

std::variant<
	weif::af::point<value_type>,
	weif::af::annular<value_type>,
	weif::af::circular<value_type>,
	weif::af::angle_averaged<value_type>,
	dimm_aperture_function<weif::af::point<value_type>>,
	dimm_aperture_function<weif::af::annular<value_type>>,
	dimm_aperture_function<weif::af::circular<value_type>>
> make_aperture_filter(value_type aperture_scale, value_type central_obscuration, bool square, const std::optional<value_type>& base_ratio) {

	if (!base_ratio) {
		if (aperture_scale == 0) {
			return weif::af::point<value_type>{};
		}

		if (square) {
			return weif::af::angle_averaged<value_type>{weif::af::square<value_type>{}, 1024};
		}

		if (central_obscuration != 0) {
			return weif::af::annular<value_type>{central_obscuration};
		}

		return weif::af::circular<value_type>{};
	} else {
		if (aperture_scale == 0) {
			return dimm_aperture_function{weif::af::point<value_type>{}, *base_ratio};
		}

		if (central_obscuration != 0) {
			return dimm_aperture_function{weif::af::annular<value_type>{central_obscuration}, *base_ratio};
		}

		return dimm_aperture_function{weif::af::circular<value_type>{}, *base_ratio};
	}
}

std::pair<value_type, std::variant<weif::sf::mono<value_type>, weif::sf::poly<value_type>>>
make_spectral_filter(const std::vector<std::string>& response_filename, std::optional<value_type> mono, std::optional<value_type> carrier) {
	if (mono) {
		return {*mono, weif::sf::mono<value_type>{}};
	}

	auto sr = weif::spectral_response<value_type>::stack_from_files(response_filename.cbegin(), response_filename.cend());
	std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
	sr.normalize();

	auto sf = [&] () {
		if (carrier)
			return weif::sf::poly{sr, 4096, *carrier};

		return weif::sf::poly{sr, 4096};
	} ();
	const auto lambda = sf.equiv_lambda();
	std::cerr << "Equivalent lambda: " << lambda << std::endl;
	sf.normalize();

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
		("base_ratio", po::value<value_type>(), "Base to aperture scale ratio")
		("central_obscuration", po::value<value_type>()->default_value(0.0), "Central obscuration")
		("output_filename", po::value<std::string>()->default_value("wf.dat"), "Output filename")
		("response_filename", po::value<std::vector<std::string>>()->required(), "Spectral response input filename")
		("square", "Use square aperture filter")
		("carrier", po::value<value_type>(), "Carrier wavelength")
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
		const std::optional<value_type> base_ratio{
			va.count("base_ratio") ? std::optional(va["base_ratio"].as<value_type>()) : std::nullopt};
		const auto central_obscuration = va["central_obscuration"].as<value_type>();
		const auto output_filename = va["output_filename"].as<std::string>();
		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();
		const bool square = va.count("square");
		const std::optional<value_type> carrier{
			va.count("carrier") ? std::optional(va["carrier"].as<value_type>()) : std::nullopt};
		const std::optional<value_type> mono{
			va.count("mono") ? std::optional(va["mono"].as<value_type>()) : std::nullopt};

		const auto [lambda, spectral_filter] = make_spectral_filter(response_filename, mono, carrier);
		const auto aperture_filter = make_aperture_filter(aperture_scale, central_obscuration, square, base_ratio);

		const xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(30), size);

		const auto t1 = std::chrono::high_resolution_clock::now();

		constexpr auto wf_grid_size = 1024 + 1;
		const auto wf = std::visit([&] (const auto& af) {
			return std::visit([&] (const auto& sf) {
				return weif::weight_function<value_type>{sf, lambda, af, aperture_scale, wf_grid_size};
			}, spectral_filter);
		}, aperture_filter);

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

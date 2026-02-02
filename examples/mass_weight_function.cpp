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

#include <weif/af/circular.h>
#include <weif/sf/poly.h>
#include <weif/spectral_response.h>
#include <weif/weight_function.h>


using value_type = float;


std::pair<value_type, weif::sf::poly<value_type>>
make_spectral_filter(const std::vector<std::string>& response_filename) {
	auto sr = weif::spectral_response<value_type>::stack_from_files(response_filename.cbegin(), response_filename.cend());
	std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
	sr.normalize();

	weif::sf::poly sf{sr, 4096};
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
		("magnification", po::value<value_type>()->default_value(16.20), "Magnification ratio")
		("output_filename", po::value<std::string>()->default_value("weights.dat"), "Output filename")
		("response_filename", po::value<std::vector<std::string>>()->required(), "Spectral response input filename");

	try {
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);

		if (va.count("help")) {
			std::cerr << opts << std::endl;

			return 1;
		}

		po::notify(va);

		const auto size = va["size"].as<std::size_t>();
		const auto magnification = va["magnification"].as<value_type>();
		const auto output_filename = va["output_filename"].as<std::string>();
		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();

		constexpr std::array<float, 4> inner = {0.00, 1.30, 2.20, 3.90};
		constexpr std::array<float, 4> outer = {1.27, 2.15, 3.85, 5.50};
		constexpr auto wf_grid_size = 1024 + 1;

		const auto [lambda, spectral_filter] = make_spectral_filter(response_filename);

		const xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(30), size);

		std::vector<weif::weight_function<value_type>> wf;
		wf.reserve(10);

		for (std::size_t i = 0; i < inner.size(); ++i) {
			for (std::size_t j = 0; j <= i; ++j) {
				const auto d1 = outer[i];
				const auto eps1 = inner[i] / outer[i];

				const auto d2 = outer[j];
				const auto eps2 = inner[j] / outer[j];

				const auto aperture_filter = weif::af::cross_annular{d2 / d1, eps1, eps2};
				wf.emplace_back(spectral_filter, lambda, aperture_filter, d1 * magnification, wf_grid_size);
			}
		}

		std::ofstream stm(output_filename);
		xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid,
			wf[0](grid),
			wf[1](grid),
			wf[2](grid),
			wf[3](grid),
			wf[4](grid),
			wf[5](grid),
			wf[6](grid),
			wf[7](grid),
			wf[8](grid),
			wf[9](grid)))));

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

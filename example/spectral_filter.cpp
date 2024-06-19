/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <fstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp> // IWYU pragma: keep

#include <xtensor/xarray.hpp> // IWYU pragma: keep
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xmanipulation.hpp>

#include <weif/sf/poly.h>


int main(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size")
		("normalize", "Normalize the filter")
		("response_filename", po::value<std::string>(), "Spectral response input filename")
		("filter_filename", po::value<std::string>(), "Spectral filter output filename");

	pos_opts.add("response_filename", 1);
	pos_opts.add("filter_filename", 1);

	try {
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);
		po::notify(va);

		const auto response_filename = va["response_filename"].as<std::string>();
		const auto filter_filename   = va["filter_filename"].as<std::string>();
		const auto size = va["size"].as<std::size_t>();

		auto sr = weif::spectral_response<float>::make_from_file(response_filename);
		sr.normalize();
		std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
		weif::sf::poly sf{sr, size};
		std::cerr << "Equivalent lambda: " << sf.equiv_lambda() << std::endl;
		if (va.count("normalize")) {
			sf.normalize();
			std::cerr << "Equivalent lambda: " << sf.equiv_lambda() << std::endl;
		}

		xt::xarray<float> grid = xt::linspace(static_cast<float>(0), static_cast<float>(5), size);

		std::ofstream stm(filter_filename);
		xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid, sf(xt::square(grid)), sf.regular(xt::square(grid))))));

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

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


using value_type = float;

int main(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size")
		("normalize", "Normalize the filter")
		("carrier", po::value<value_type>(), "Carrier wavelength")
		("response_filename", po::value<std::vector<std::string>>()->required(), "Spectral response input filename")
		("filter_filename", po::value<std::string>(), "Spectral filter output filename");

	pos_opts.add("filter_filename", 1);

	try {
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);
		po::notify(va);

		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();
		const auto filter_filename   = va["filter_filename"].as<std::string>();
		const auto size = va["size"].as<std::size_t>();
		const std::optional<value_type> carrier{va.count("carrier") ? std::optional(va["carrier"].as<value_type>()) : std::nullopt};

		auto sr = weif::spectral_response<value_type>::stack_from_files(response_filename.cbegin(), response_filename.cend());
		sr.normalize();
		std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
		auto sf = [&] () {
			if (carrier)
				return weif::sf::poly{sr, size, *carrier};

			return weif::sf::poly{sr, size};
		} ();
		std::cerr << "Equivalent lambda: " << sf.equiv_lambda() << std::endl;
		std::cerr << "Carrier lambda:    " << sf.carrier() << std::endl;
		if (va.count("normalize")) {
			sf.normalize();
			std::cerr << "Equivalent lambda: " << sf.equiv_lambda() << std::endl;
			std::cerr << "Carrier lambda:    " << sf.carrier() << std::endl;
		}

		xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(5), size);

		std::ofstream stm(filter_filename);
		xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid, sf(xt::square(grid)), sf.regular(xt::square(grid))))));

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

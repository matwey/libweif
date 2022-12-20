#include <chrono>
#include <fstream>
#include <iostream>
#include <variant>
#include <vector>

#include <boost/program_options.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xmanipulation.hpp>

#include <aperture_filter.h>
#include <spectral_filter.h>
#include <spectral_response.h>
#include <weight_function_grid_2d.h>


using value_type = float;

std::variant<weif::af::point<value_type>, weif::af::annular<value_type>, weif::af::circular<value_type>>
make_aperture_filter(value_type aperture_scale, value_type central_obscuration) {
	if (aperture_scale == 0) {
		return weif::af::point<value_type>{};
	}

	if (central_obscuration != 0) {
		return weif::af::annular<value_type>{central_obscuration};
	}

	return weif::af::circular<value_type>{};
}

std::pair<value_type, std::variant<weif::mono_spectral_filter<value_type>, weif::spectral_filter<value_type>>>
make_spectral_filter(const std::vector<std::string>& response_filename, bool mono) {
	auto response_it = response_filename.cbegin();
	auto sr = weif::spectral_response<value_type>::make_from_file(*response_it++);
	for (; response_it != response_filename.cend(); ++response_it) {
		auto sr2 = weif::spectral_response<value_type>::make_from_file(*response_it);
		sr.stack(sr2);
	}

	std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
	sr.normalize();

	weif::spectral_filter sf{sr, 4096};
	const auto lambda = sf.equiv_lambda();
	std::cerr << "Equivalent lambda: " << lambda << std::endl;
	sf.normalize();

	if (mono) {
		return {lambda, weif::mono_spectral_filter<value_type>{}};
	}

	return {lambda, std::move(sf)};
}


int main(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("aperture_scale", po::value<value_type>()->default_value(11), "Aperture scale, mm.")
		("central_obscuration", po::value<value_type>()->default_value(0.0), "Central obscuration")
		("grid_step", po::value<value_type>()->default_value(11), "Grid step, mm.")
		("grid_size", po::value<std::size_t>()->default_value(121), "Grid size")
		("output_filename", po::value<std::string>()->default_value("wf.dat"), "Output filename")
		("response_filename", po::value<std::vector<std::string>>()->required(), "Spectral response input filename")
		("mono", "Use monochromatic spectral filter instead of polychromatic one");

	try {
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);

		if (va.count("help")) {
			std::cerr << opts << std::endl;

			return 1;
		}

		po::notify(va);

		const auto aperture_scale = va["aperture_scale"].as<value_type>();
		const auto central_obscuration = va["central_obscuration"].as<value_type>();
		const auto grid_step = va["grid_step"].as<value_type>();
		const auto grid_size = va["grid_size"].as<std::size_t>();
		const auto output_filename = va["output_filename"].as<std::string>();
		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();
		const bool mono = va.count("mono");

		const auto [lambda, spectral_filter] = make_spectral_filter(response_filename, mono);
		const auto aperture_filter = make_aperture_filter(aperture_scale, central_obscuration);

		const xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(30), 1024);

		const auto t1 = std::chrono::high_resolution_clock::now();

		const auto wf = std::visit([&] (const auto& af) {
			return std::visit([&] (const auto& sf) {
				return weif::weight_function_grid_2d<value_type>{sf, lambda, af, aperture_scale, grid_step, std::array{static_cast<std::size_t>(grid_size), static_cast<std::size_t>(grid_size)}};
			}, spectral_filter);
		}, aperture_filter);

		const auto t2 = std::chrono::high_resolution_clock::now();

		std::ofstream stm(output_filename);
		xt::dump_csv(stm, wf(2.0));

		std::cerr << "Consumed time: " << std::chrono::duration_cast<std::chrono::duration<value_type>>(t2-t1).count() << " sec" << std::endl;

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

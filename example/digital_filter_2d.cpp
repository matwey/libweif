#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xmanipulation.hpp>

#include <aperture_filter.h>
#include <digital_filter_2d.h>
#include <spectral_filter.h>
#include <weight_function.h>


using value_type = float;

std::pair<value_type, weif::spectral_filter<value_type>>
make_spectral_filter(const std::vector<std::string>& response_filename) {
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

	return {lambda, std::move(sf)};
}

int main(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size")
		("impulse_size", po::value<std::size_t>()->default_value(121), "Filter impulse size")
		("aperture_scale", po::value<value_type>()->default_value(11), "Aperture scale, mm.")
		("output_filename", po::value<std::string>()->default_value("wf.dat"), "Output filename")
		("response_filename", po::value<std::vector<std::string>>()->required(), "Spectral response input filename");

	try {
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);
		po::notify(va);

		const auto size = va["size"].as<std::size_t>();
		const auto impulse_size = va["impulse_size"].as<std::size_t>();
		const auto aperture_scale = va["aperture_scale"].as<value_type>();
		const auto output_filename = va["output_filename"].as<std::string>();
		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();

		const auto [lambda, sf] = make_spectral_filter(response_filename);

		const xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(30), size);

		const auto t1 = std::chrono::high_resolution_clock::now();

		const weif::square_aperture<value_type> square_af{};
		const weif::digital_filter_2d<value_type> df{[&square_af](value_type ux, value_type uy) noexcept {
			using namespace std;

			const auto u2 = ux * ux + uy * uy;

			return pow(u2 * 4, static_cast<value_type>(5.0/6.0)) / square_af(ux, uy);
		}, std::array{impulse_size, impulse_size}};
		const weif::angle_averaged<value_type> af{[&square_af, &df](value_type ux, value_type uy) noexcept {
			return square_af(ux, uy) * df(ux, uy);
		}, 1024};
		constexpr auto wf_grid_size = 1024 + 1;
		const weif::weight_function<value_type> wf{sf, lambda, af, aperture_scale, wf_grid_size};

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

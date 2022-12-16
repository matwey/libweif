#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xmanipulation.hpp>

#include <digital_filter_2d.h>


using value_type = float;

int main(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size")
		("impulse_size", po::value<std::size_t>()->default_value(121), "Filter impulse size")
		("impulse_filename", po::value<std::string>(), "Digital filter impulse output filename")
		("filter_filename", po::value<std::string>(), "Digital filter output filename");

	pos_opts.add("impulse_filename", 1);
	pos_opts.add("filter_filename", 1);

	try {
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);
		po::notify(va);

		const auto filter_filename = va["filter_filename"].as<std::string>();
		const auto impulse_filename = va["impulse_filename"].as<std::string>();
		const auto size = va["size"].as<std::size_t>();
		const auto impulse_size = va["impulse_size"].as<std::size_t>();

		const weif::digital_filter_2d<value_type> df{[](value_type ux, value_type uy) noexcept {
			using namespace std;

			const auto u2 = ux * ux + uy * uy;

			return pow(u2 * 4, static_cast<value_type>(5.0/6.0));
		}, std::array{impulse_size, impulse_size}};

		xt::xarray<float> grid = xt::linspace(static_cast<float>(0), static_cast<float>(1), size);

		{
			std::ofstream stm(impulse_filename);
			xt::dump_csv(stm, df.impulse());
		}

		{
			std::ofstream stm(filter_filename);
			xt::dump_csv(stm, df(grid, grid));
		}

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

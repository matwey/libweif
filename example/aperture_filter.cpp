#include <iostream>
#include <fstream>
#include <string>

#include <boost/program_options.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xmanipulation.hpp>

#include <aperture_filter.h>

template<class AF>
void dump_aperture_filter(const std::string& filename, const AF& af, std::size_t size) {
	xt::xarray<float> grid = xt::linspace(static_cast<float>(0), static_cast<float>(5), size);

	std::ofstream stm(filename);
	xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid, af(grid)))));
}


int main(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size");

	try {
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);
		po::notify(va);

		dump_aperture_filter("circular_aperture.csv", weif::af::circular<float>{}, va["size"].as<std::size_t>());
		dump_aperture_filter("annular_aperture.csv", weif::af::annular<float>{0.25}, va["size"].as<std::size_t>());

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

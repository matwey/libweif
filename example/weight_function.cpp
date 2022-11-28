#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xmanipulation.hpp>

#include <aperture_filter.h>
#include <spectral_filter.h>
#include <spectral_response.h>
#include <weight_function.h>


int main(int argc, char** argv) {
	namespace po = boost::program_options;

	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size")
		("response_filename", po::value<std::string>(), "Spectral response input filename");

	pos_opts.add("response_filename", 1);

	try {
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);
		po::notify(va);

		const auto response_filename = va["response_filename"].as<std::string>();
		const auto size = va["size"].as<std::size_t>();

		auto sr = weif::spectral_response<float>::make_from_file(response_filename);
		sr.normalize();
		weif::spectral_filter sf{sr, 4096};
		const auto lambda = sf.equiv_lambda();
		sf.normalize();

		weif::weight_function<float> wf{sf, lambda, weif::circular_aperture<float>{}, 11, 1025};

		xt::xarray<float> grid = xt::linspace(static_cast<float>(0), static_cast<float>(30), size);

		std::ofstream stm("wf.dat");
		xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid, wf(grid)))));

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

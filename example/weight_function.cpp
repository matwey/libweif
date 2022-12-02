#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

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
		("aperture_scale", po::value<float>()->default_value(20.574), "Aperture scale, mm.")
		("central_obscuration", po::value<float>()->default_value(0.0), "Central obscuration")
		("output_filename", po::value<std::string>()->default_value("wf.dat"), "Output filename")
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
		const auto aperture_scale = va["aperture_scale"].as<float>();
		const auto central_obscuration = va["central_obscuration"].as<float>();
		const auto output_filename = va["output_filename"].as<std::string>();
		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();

		auto response_it = response_filename.cbegin();
		auto sr = weif::spectral_response<float>::make_from_file(*response_it++);
		for (; response_it != response_filename.cend(); ++response_it) {
			auto sr2 = weif::spectral_response<float>::make_from_file(*response_it);
			sr.stack(sr2);
		}

		std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
		sr.normalize();

		const auto t1 = std::chrono::high_resolution_clock::now();

		weif::spectral_filter sf{sr, 4096};
		const auto lambda = sf.equiv_lambda();
		std::cerr << "Equivalent lambda: " << lambda << std::endl;
		sf.normalize();

		const xt::xarray<float> grid = xt::linspace(static_cast<float>(0), static_cast<float>(30), size);

		const auto wf = (central_obscuration != 0 ?
			weif::weight_function<float>{sf, lambda, weif::annular_aperture<float>{central_obscuration}, aperture_scale, 1025} :
			weif::weight_function<float>{sf, lambda, weif::circular_aperture<float>{}, aperture_scale, 1025});

		const auto t2 = std::chrono::high_resolution_clock::now();

		std::ofstream stm(output_filename);
		xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid, wf(grid)))));

		std::cerr << "Consumed time: " << std::chrono::duration_cast<std::chrono::duration<float>>(t2-t1).count() << " sec" << std::endl;

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

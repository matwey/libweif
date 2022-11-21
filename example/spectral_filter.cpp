#include <iostream>

#include <xtensor/xio.hpp>
#include <boost/program_options.hpp>

#include <spectral_filter.h>


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

		auto sr = weif::spectral_response<float>::make_from_file(response_filename);
		sr.normalize();
		std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
		weif::spectral_filter sf{sr, va["size"].as<std::size_t>()};
		std::cerr << "Equivalent lambda: " << sf.equiv_lambda() << std::endl;
		if (va.count("normalize")) {
			sf.normalize();
			std::cerr << "Equivalent lambda: " << sf.equiv_lambda() << std::endl;
		}

		sf.dump(filter_filename);

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

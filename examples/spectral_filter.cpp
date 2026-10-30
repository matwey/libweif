/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

/**
 * @page tutorial_spectral_filter Spectral response and filter
 *
 * @tableofcontents
 *
 * @section overview Overview
 *
 * This tutorial demonstrates how to use the `libweif` library to process
 * spectral response curves and compute polychromatic spectral filters. The
 * example program `spectral_filter` loads one or more spectral response files,
 * stacks them (multiplies), and evaluates the corresponding polychromatic
 * filter over a frequency grid.
 *
 * The `weif::sf::poly` class implements polychromatic spectral filter using
 * numerically provided spectral response data (e.g., filter transmission,
 * detector sensitivity, or source spectrum).
 *
 * For more details, see the complete source code: \ref spectral_filter.cpp
 *
 * @section numeric_type Choosing the numeric type
 *
 * The example uses `float` as the numeric type for calculations. You can
 * change it to `double` for higher precision if needed. The type alias
 * `value_type` is defined at the beginning of the program.
 *
 * \snippet{lineno,trimleft} spectral_filter.cpp Define value_type
 *
 * @section command_line Command‑line interface
 *
 * At the beginning of the program, the command line options are defined using
 * Boost.Program_options. Using Boost.Program_options is generally not
 * required; this part is specific to this particular example program.
 *
 * \snippet{lineno,trimleft} spectral_filter.cpp Define command line options
 *
 * We obtain the following command line interface:
 * \code{bash}
 * > spectral_filter --help
 * Allowed options:
 * --help                  Print help message
 * --size arg (=1024)      Output grid size
 * --normalize             Normalize the filter
 * --carrier arg           Carrier wavelength
 * --response_filename arg Spectral response input filename
 * --filter_filename arg   Spectral filter output filename
 *
 * \endcode
 *
 * @section parsing Parsing command‑line options
 *
 * The program uses Boost.Program_options to parse the command line. The parsed
 * values are stored in a `variables_map` and later extracted into appropriate
 * variables.
 *
 * \snippet{lineno,trimleft} spectral_filter.cpp Parse command line options
 *
 * We will explain every variable later in this tutorial.
 *
 * @section loading Loading and stacking spectral responses
 *
 * Spectral response curves are loaded from plain text files containing two
 * columns: wavelength (nm) and response value (arbitrary units). Here,
 * `response_filename` is a vector of filenames, so multiple files can be
 * provided. The responses are stacked (multiplied) over their overlapping
 * wavelength range using a helper function.
 *
 * \snippet{lineno,trimleft} spectral_filter.cpp Load and stack spectral response
 *
 * The second line is particularly important; it performs spectral response
 * normalization. After normalization, the spectral response has unit area
 * under its curve. Without normalization, the resulting polychromatic filter
 * will have incorrect amplitude, resulting in incorrect weight functions.
 *
 * The last line prints the effective wavelength of the stacked spectral
 * response. When everything is done correctly, the result is a reasonable
 * value of about hundreds of nanometers.
 *
 * @section filter Creating the polychromatic filter
 *
 * The polychromatic filter `weif::sf::poly` is constructed from the spectral
 * response. You may optionally specify a carrier wavelength; if omitted, the
 * effective wavelength of the response is used as the carrier. Normally, you
 * don't need to care about the carrier wavelength. It is a numerical parameter
 * whose value affects the accuracy of the result rather than the physical
 * meaning. `size` is the number of internal nodes used for interpolation.
 * With the default carrier wavelength, `1024` should be sufficient, but you
 * can safely experiment with higher values.
 *
 * \snippet{lineno,trimleft} spectral_filter.cpp Create spectral filter
 *
 * The constructor internally computes the Fourier transform of the spectral
 * response and stores it for fast evaluation. Then both equivalent
 * wavelength—the monochromatic wavelength that would produce the same
 * scintillation with the point aperture—and carrier wavelength are printed for
 * reference.
 *
 * @section normalization Spectral filter normalization
 *
 * If the `--normalize` flag is given, the filter is normalized. Normalization
 * changes the equivalent and carrier wavelengths, which are printed to stderr
 * for verification. Normally, you should always normalize the polychromatic
 * filter and provide the spectral scale (equivalent wavelength) separately.
 *
 * \snippet{lineno,trimleft} spectral_filter.cpp Normalize spectral filter
 *
 * @section output Generating output data
 *
 * The program evaluates the filter on a uniform grid of squared frequencies.
 * The grid spans from 0 to 5 in dimensionless units if you performed filter
 * normalization.
 *
 * \snippet{lineno,trimleft} spectral_filter.cpp Generate output
 *
 * The output CSV file contains two columns:
 * 1. The grid coordinate \f$u\f$
 * 2. The filter value \f$E(u)\f$
 *
 * If you avoid normalization, the output units are more complicated. The grid
 * coordinate \f$ u = z f^2 \f$ is expressed in units of \f$ \mathrm{nm}^{-1}
 * \f$.
 */

/**
 * @example spectral_filter.cpp
 *
 * This is an example of how to use the \ref weif::spectral_response and \ref
 * weif::sf::poly classes.
 */

#include <fstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp> // IWYU pragma: keep

#include <xtensor/containers/xarray.hpp> // IWYU pragma: keep
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/io/xcsv.hpp>
#include <xtensor/misc/xmanipulation.hpp>

#include <weif/sf/poly.h>


//! [Define value_type]
using value_type = float;
//! [Define value_type]

int main(int argc, char** argv) {
//! [Define command line options]
	namespace po = boost::program_options;

	po::options_description opts("Allowed options");
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("help", "Print help message")
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size")
		("normalize", "Normalize the filter")
		("carrier", po::value<value_type>(), "Carrier wavelength")
		("response_filename", po::value<std::vector<std::string>>()->required(), "Spectral response input filename")
		("filter_filename", po::value<std::string>(), "Spectral filter output filename");

	pos_opts.add("filter_filename", 1);
//! [Define command line options]

	try {
//! [Parse command line options]
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);

		if (va.count("help")) {
			std::cout << opts << std::endl;
			return 1;
		}

		po::notify(va);

		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();
		const auto filter_filename   = va["filter_filename"].as<std::string>();
		const auto size = va["size"].as<std::size_t>();
		const std::optional<value_type> carrier{va.count("carrier") ? std::optional(va["carrier"].as<value_type>()) : std::nullopt};
//! [Parse command line options]

//! [Load and stack spectral response]
		auto sr = weif::spectral_response<value_type>::stack_from_files(response_filename.cbegin(), response_filename.cend());
		sr.normalize();
		std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
//! [Load and stack spectral response]

//! [Create spectral filter]
		auto sf = carrier ? weif::sf::poly{sr, size, *carrier} : weif::sf::poly{sr, size};
		std::cerr << "Equivalent lambda: " << sf.equiv_lambda() << std::endl;
		std::cerr << "Carrier lambda:    " << sf.carrier() << std::endl;
//! [Create spectral filter]

//! [Normalize spectral filter]
		if (va.count("normalize")) {
			sf.normalize();
			std::cerr << "Equivalent lambda: " << sf.equiv_lambda() << std::endl;
			std::cerr << "Carrier lambda:    " << sf.carrier() << std::endl;
		}
//! [Normalize spectral filter]

//! [Generate output]
		xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(5), size);

		std::ofstream stm(filter_filename);
		xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid, sf(xt::square(grid)), sf.regular(xt::square(grid))))));
//! [Generate output]

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

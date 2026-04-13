/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

/**
 * @page tutorial_mass_weight_function MASS weight function calculation
 *
 * @tableofcontents
 *
 * @section overview Overview
 *
 * This tutorial demonstrates how to use the `libweif` library to compute the
 * complete set of weight functions for a Multi‑Aperture Scintillation Sensor
 * (MASS) instrument (Tokovinin et al. (2003) "Restoration of turbulence profile from scintillation indices",
 * https://doi.org/10.1046/j.1365-8711.2003.06731.x). The example program
 * `mass_weight_function` calculates the ten weight functions that describe the
 * variance of relative flux fluctuations in each of the four annular apertures
 * (A, B, C, D) of a MASS instrument and the covariances between them.
 *
 * For more details, see the complete source code: \ref mass_weight_function.cpp
 *
 * @section numeric_type Choosing the numeric type
 *
 * The example uses `float` as the numeric type for calculations. The type alias
 * `value_type` is defined at the beginning of the program.
 *
 * \snippet{lineno,trimleft} mass_weight_function.cpp Define value_type
 *
 * @section command_line Command‑line interface
 *
 * The program accepts a few command‑line arguments that control the output
 * resolution, the optical magnification, and the spectral response. The options
 * are defined using Boost.Program_options.
 *
 * \snippet{lineno,trimleft} mass_weight_function.cpp Define command line options
 *
 * We obtain the following command line interface:
 * \code{bash}
 * > mass_weight_function --help
 * --help                               Print help message
 * --size arg (=1024)                   Output grid size
 * --magnification arg (=16.2000008)    Magnification ratio
 * --output_filename arg (=weights.dat) Output filename
 * --response_filename arg              Spectral response input filename
 * \endcode
 *
 * @section parsing Parsing command‑line options
 *
 * The program uses Boost.Program_options to parse the command line. The parsed
 * values are stored in a `variables_map` and later extracted into appropriate
 * variables.
 *
 * \snippet{lineno,trimleft} mass_weight_function.cpp Parse command line options
 *
 * @section spectral_filter Creating the spectral filter
 *
 * A polychromatic spectral filter is created from one or more spectral response
 * files. Typically, the spectral response combines the source star spectrum and
 * the detector's spectral sensitivity. The responses are loaded, stacked,
 * normalized, and then used to construct a `weif::sf::poly` filter. The
 * equivalent wavelength is stored before normalization, because normalization
 * sets the filter amplitude to unity.
 *
 * \snippet{lineno,trimleft} mass_weight_function.cpp Create spectral filter
 *
 * The helper function `make_spectral_filter` returns both the equivalent
 * wavelength and the filter object.
 *
 * @section mass_apertures MASS aperture geometry
 *
 * The four MASS apertures are defined by inner and outer radii of the
 * segmentator. The example uses the standard MASS‑DIMM aperture set:
 * | Aperture | Inner radius (mm) | Outer radius (mm) |
 * | -------: | :---------------: | :---------------: |
 * | A        |       \f$0.00\f$  |       \f$1.27\f$  |
 * | B        |       \f$1.30\f$  |       \f$2.15\f$  |
 * | C        |       \f$2.20\f$  |       \f$3.85\f$  |
 * | D        |       \f$3.90\f$  |       \f$5.50\f$  |
 *
 * These radii are scaled by the dimensionless magnification factor
 * (command‑line argument `--magnification`) to obtain the physical aperture
 * sizes at the input pupil.  We use \f$16.2\f$ as the default magnification.
 *
 * \snippet{lineno,trimleft} mass_weight_function.cpp Define MASS apertures
 *
 * @section weight_functions Constructing the weight functions
 *
 * First, the spectral filter is prepared using the parsed command‑line
 * arguments.
 *
 * \snippet{lineno,trimleft} mass_weight_function.cpp Prepare spectral filter
 *
 * For each pair of apertures (including each aperture with itself), a
 * cross‑annular aperture filter is created. The filter parameters are derived
 * from the relative sizes and obscuration ratios of the two apertures. Then a
 * `weif::weight_function` object is instantiated with the common spectral
 * filter, the equivalent wavelength, the cross‑annular aperture filter, and the
 * scaled aperture size.
 *
 * \snippet{lineno,trimleft} mass_weight_function.cpp Create weight functions
 *
 * There are ten such weight functions, corresponding to the ten independent
 * elements of the covariance matrix (A, AB, AC, AD, B, BC, BD, C, CD, D).
 *
 * @section evaluation Evaluating the weight functions
 *
 * All ten weight functions are evaluated on the same uniform altitude grid
 * spanning from 0 to 30 km. The results are written to a single CSV file with
 * eleven columns: the altitude followed by the ten weight function values.
 * We use C++20 features to make the code more concise and avoid manually
 * enumerating elements.
 *
 * \snippet{lineno,trimleft} mass_weight_function.cpp Evaluate and output
 *
 * @section output Generated output
 *
 * After a successful run, a CSV file (default `weights.dat`) is created
 * containing eleven columns:
 * 1. The altitude \f$h\f$ in kilometers
 * 2. Weight function A (\f$W_{A}(h)\f$)
 * 3. Weight function AB (\f$W_{AB}(h)\f$)
 * 4. Weight function AC (\f$W_{AC}(h)\f$)
 * 5. Weight function AD (\f$W_{AD}(h)\f$)
 * 6. Weight function B (\f$W_{B}(h)\f$)
 * 7. Weight function BC (\f$W_{BC}(h)\f$)
 * 8. Weight function BD (\f$W_{BD}(h)\f$)
 * 9. Weight function C (\f$W_{C}(h)\f$)
 * 10. Weight function CD (\f$W_{CD}(h)\f$)
 * 11. Weight function D (\f$W_{D}(h)\f$)
 *
 * The weight functions are proportional to the covariance of scintillation
 * between the corresponding apertures. Integrating \f$W_{ij}(h) C_n^2(h) dh\f$
 * over altitude yields the covariance element \f$\langle s_i s_j \rangle\f$.
 *
 * @section units Consistent units in the library
 *
 * The library uses the following consistent units:
 * - Altitudes: kilometers (km)
 * - Wavelengths: nanometers (nm)
 * - Geometric scales: millimeters (mm)
 *
 * The example respects these conventions; the aperture radii are given in
 * millimeters and then they are multiplied by the magnification (dimensionless)
 * to obtain the physical aperture size on the input pupil. The output altitudes
 * are in kilometers, and the weight function values are in units of
 * \f$\mathrm{m}^{-1/3}\f$.
 */

/**
 * @example mass_weight_function.cpp
 *
 * This is an example of how to compute the complete set of weight functions for
 * a Multi‑Aperture Scintillation Sensor (MASS) instrument.
 */

#include <array>
#include <fstream>
#include <iostream>
#include <utility>
#include <variant>
#include <vector>

#include <boost/program_options.hpp>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/io/xcsv.hpp>
#include <xtensor/misc/xmanipulation.hpp>

#include <weif/af/circular.h>
#include <weif/sf/poly.h>
#include <weif/spectral_response.h>
#include <weif/weight_function.h>


//! [Define value_type]
using value_type = float;
//! [Define value_type]

//! [Create spectral filter]
std::pair<value_type, weif::sf::poly<value_type>>
make_spectral_filter(const std::vector<std::string>& response_filename) {
	auto sr = weif::spectral_response<value_type>::stack_from_files(response_filename.cbegin(), response_filename.cend());
	std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
	sr.normalize();

	weif::sf::poly sf{sr, 4096};
	const auto lambda = sf.equiv_lambda();
	std::cerr << "Equivalent lambda: " << lambda << std::endl;
	sf.normalize();

	return {lambda, std::move(sf)};
}
//! [Create spectral filter]

int main(int argc, char** argv) {
	namespace po = boost::program_options;

//! [Define command line options]
	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("help", "Print help message")
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size")
		("magnification", po::value<value_type>()->default_value(16.20), "Magnification ratio")
		("output_filename", po::value<std::string>()->default_value("weights.dat"), "Output filename")
		("response_filename", po::value<std::vector<std::string>>()->required(), "Spectral response input filename");
//! [Define command line options]

	try {
//! [Parse command line options]
		auto parsed = po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run();
		po::store(std::move(parsed), va);

		if (va.count("help")) {
			std::cerr << opts << std::endl;

			return 1;
		}

		po::notify(va);

		const auto size = va["size"].as<std::size_t>();
		const auto magnification = va["magnification"].as<value_type>();
		const auto output_filename = va["output_filename"].as<std::string>();
		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();
//! [Parse command line options]

//! [Define MASS apertures]
		constexpr std::array<value_type, 4> inner = {0.00, 1.30, 2.20, 3.90};
		constexpr std::array<value_type, 4> outer = {1.27, 2.15, 3.85, 5.50};
//! [Define MASS apertures]

//! [Prepare spectral filter]
		const auto [lambda, spectral_filter] = make_spectral_filter(response_filename);
//! [Prepare spectral filter]

//! [Create weight functions]
		constexpr auto wf_grid_size = 1024 + 1;
		std::vector<weif::weight_function<value_type>> wf;
		wf.reserve(10);

		for (std::size_t i = 0; i < inner.size(); ++i) {
			for (std::size_t j = 0; j <= i; ++j) {
				const auto d1 = outer[i];
				const auto eps1 = inner[i] / outer[i];

				const auto d2 = outer[j];
				const auto eps2 = inner[j] / outer[j];

				const auto aperture_filter = weif::af::cross_annular{d2 / d1, eps1, eps2};
				wf.emplace_back(spectral_filter, lambda, aperture_filter, d1 * magnification, wf_grid_size);
			}
		}
//! [Create weight functions]

//! [Evaluate and output]
		const xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(30), size);

		auto make_wf_tuple = [&grid, &wf]<std::size_t... Is>(std::index_sequence<Is...>) {
			return xt::xtuple(grid, wf[Is](grid)...);
		};

		std::ofstream stm(output_filename);
		xt::dump_csv(stm, xt::transpose(xt::vstack(make_wf_tuple(std::make_index_sequence<10>{}))));
//! [Evaluate and output]

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

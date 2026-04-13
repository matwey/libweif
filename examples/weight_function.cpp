/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

/**
 * @page tutorial_weight_function Weight function calculation
 *
 * @tableofcontents
 *
 * @section overview Overview
 *
 * This tutorial demonstrates how to use the `libweif` library to compute weight
 * functions for atmospheric turbulence scintillation. The example program
 * `weight_function` combines aperture and spectral filters to evaluate the
 * weight function—the contribution of turbulence layers at different altitudes
 * to the observed scintillation variance.
 *
 * Weight functions integrate the Kolmogorov turbulence spectrum over spatial
 * frequencies, taking into account the optical system's aperture geometry and
 * spectral response.
 *
 * The `weif::weight_function` class encapsulates this calculation, accepting a
 * spectral filter, a wavelength, an aperture filter, and an aperture scale.
 *
 * For more details, see the complete source code: \ref weight_function.cpp
 *
 * @section numeric_type Choosing the numeric type
 *
 * The example uses `float` as the numeric type for calculations. You can change
 * it to `double` for higher precision if needed. The type alias `value_type` is
 * defined at the beginning of the program.
 *
 * \snippet{lineno,trimleft} weight_function.cpp Define value_type
 *
 * @section command_line Command‑line interface
 *
 * The program accepts several command‑line arguments that control the aperture
 * geometry, spectral response, and output parameters. The options are defined
 * using Boost.Program_options.
 *
 * \snippet{lineno,trimleft} weight_function.cpp Define command line options
 *
 * We obtain the following command line interface:
 * \code{bash}
 * > weight_function --help
 * Allowed options:
 * --size arg (=1024)                 Output grid size
 * --aperture_scale arg (=20.5739994) Aperture scale, mm.
 * --central_obscuration arg (=0)     Central obscuration
 * --output_filename arg (=wf.dat)    Output filename
 * --response_filename arg            Spectral response input filename
 * --square                           Use square aperture filter
 * --carrier arg                      Carrier wavelength
 * --mono arg                         Use monochromatic spectral filter with
 *                                    given wavelength
 * \endcode
 *
 * @section parsing Parsing command‑line options
 *
 * The program uses Boost.Program_options to parse the command line. The parsed
 * values are stored in a `variables_map` and later extracted into appropriate
 * variables.
 *
 * \snippet{lineno,trimleft} weight_function.cpp Parse command line options
 *
 * @section spectral_filter Creating the spectral filter
 *
 * The spectral filter can be either monochromatic (`weif::sf::mono`) or
 * polychromatic (`weif::sf::poly`). If the `--mono` option is given, a
 * monochromatic filter at the specified wavelength is used. Otherwise, one or
 * more spectral response files are loaded, stacked, and normalized to create a
 * polychromatic filter. Additionally, the `--carrier` option can be used to
 * specify the carrier wavelength, which affects only numerical precision.  This
 * example demonstrates as many options as possible.  Normally, you would create
 * a spectral filter once and not worry about the carrier wavelength.  Note that
 * the equivalent wavelength should be stored before the spectral filter is
 * normalized, because normalization sets it to unity.
 *
 * \snippet{lineno,trimleft} weight_function.cpp Create spectral filter
 *
 * The helper function `make_spectral_filter` returns both the equivalent
 * wavelength (needed for scaling) and the filter object itself.
 *
 * @section aperture_filter Creating the aperture filter
 *
 * In this example we allow the user to specify the aperture scale at runtime.
 * Normally, you would just create the aperture filter you need.  The following
 * possibilities are implemented:
 * - `weif::af::point` – if the aperture scale is zero (point aperture)
 * - `weif::af::angle_averaged<weif::af::square>` – if `--square` is given
 * - `weif::af::annular` – if a non‑zero central obscuration is specified
 * - `weif::af::circular` – otherwise (uniform circular aperture)
 *
 * Note that `weif::af::square` is used in conjunction with
 * `weif::af::angle_averaged` because weight function calculation requires
 * axisymmetric filters.
 *
 * \snippet{lineno,trimleft} weight_function.cpp Create aperture filter
 *
 * @section weight_function Constructing the weight function
 *
 * First, the spectral and aperture filters are prepared using the parsed
 * command‑line arguments.
 *
 * \snippet{lineno,trimleft} weight_function.cpp Prepare spectral and aperture filters
 *
 * Because this example allows considerable flexibility at runtime, the weight
 * function object is created by visiting both filter variants (spectral and
 * aperture) and passing them to the `weif::weight_function` constructor
 * together with the equivalent wavelength and aperture scale.  Normally, you
 * would just create the weight function you need with the required spectral and
 * geometric parameters.  An internal grid size `wf_grid_size` is used for
 * interpolation.
 *
 * \snippet{lineno,trimleft} weight_function.cpp Create weight function
 *
 * @section evaluation Evaluating and timing
 *
 * The program evaluates the weight function on a uniform altitude grid spanning
 * from \f$0\f$ to \f$30\,\mathrm{km}\f$.
 *
 * \snippet{lineno,trimleft} weight_function.cpp Evaluate weight function
 *
 * @section output Generated output
 *
 * After a successful run, a CSV file (default `wf.dat`) is created containing
 * two columns:
 * 1. The altitude \f$h\f$ in kilometers
 * 2. The weight function value \f$W(h)\f$
 *
 * The weight function is proportional to \f$C_n^2(h)\f$, the turbulence
 * strength profile. Integrating \f$W(h) C_n^2(h)\f$ over altitude \f$h\f$
 * yields the total scintillation variance.
 *
 * @section units Consistent units in the library
 *
 * The library uses the following consistent units:
 * - Altitudes: kilometers (km)
 * - Wavelengths: nanometers (nm)
 * - Geometric scales: millimeters (mm)
 *
 * The example respects these conventions; the aperture scale is provided in
 * millimeters, and the output altitudes are in kilometers. The resulting weight
 * function values are in units of \f$\mathrm{m}^{-1/3}\f$.
 * 
 */

/**
 * @example weight_function.cpp
 *
 * This is an example of how to use the \ref weif::weight_function class
 * together with aperture and spectral filters.
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <variant>
#include <vector>

#include <boost/program_options.hpp>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/io/xcsv.hpp>
#include <xtensor/misc/xmanipulation.hpp>

#include <weif/af/angle_averaged.h>
#include <weif/af/circular.h>
#include <weif/af/square.h>
#include <weif/af/point.h>
#include <weif/sf/mono.h>
#include <weif/sf/poly.h>
#include <weif/spectral_response.h>
#include <weif/weight_function.h>


//! [Define value_type]
using value_type = float;
//! [Define value_type]

//! [Create aperture filter]
std::variant<
	weif::af::point<value_type>,
	weif::af::annular<value_type>,
	weif::af::circular<value_type>,
	weif::af::angle_averaged<value_type>
> make_aperture_filter(value_type aperture_scale, value_type central_obscuration, bool square) {
	if (aperture_scale == 0) {
		return weif::af::point<value_type>{};
	} else if (square) {
		return weif::af::angle_averaged{weif::af::square<value_type>{}, 1024};
	} else if (central_obscuration != 0) {
		return weif::af::annular<value_type>{central_obscuration};
	}

	return weif::af::circular<value_type>{};
}
//! [Create aperture filter]

//! [Create spectral filter]
std::pair<value_type,
	std::variant<
		weif::sf::mono<value_type>,
		weif::sf::poly<value_type>
	>
> make_spectral_filter(const std::vector<std::string>& response_filename, std::optional<value_type> mono, std::optional<value_type> carrier) {
	if (mono) {
		return {*mono, weif::sf::mono<value_type>{}};
	}

	auto sr = weif::spectral_response<value_type>::stack_from_files(response_filename.cbegin(), response_filename.cend());
	std::cerr << "Effective lambda: " << sr.effective_lambda() << std::endl;
	sr.normalize();

	constexpr std::size_t size = 4096;
	auto sf = carrier ? weif::sf::poly{sr, size, *carrier} : weif::sf::poly{sr, size};
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
		("aperture_scale", po::value<value_type>()->default_value(20.574), "Aperture scale, mm.")
		("central_obscuration", po::value<value_type>()->default_value(0.0), "Central obscuration")
		("output_filename", po::value<std::string>()->default_value("wf.dat"), "Output filename")
		("response_filename", po::value<std::vector<std::string>>()->required(), "Spectral response input filename")
		("square", "Use square aperture filter")
		("carrier", po::value<value_type>(), "Carrier wavelength")
		("mono", po::value<value_type>(), "Use monochromatic spectral filter with given wavelength");
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
		const auto aperture_scale = va["aperture_scale"].as<value_type>();
		const auto central_obscuration = va["central_obscuration"].as<value_type>();
		const auto output_filename = va["output_filename"].as<std::string>();
		const auto response_filename = va["response_filename"].as<std::vector<std::string>>();
		const bool square = va.count("square");
		const std::optional<value_type> carrier{
			va.count("carrier") ? std::optional(va["carrier"].as<value_type>()) : std::nullopt};
		const std::optional<value_type> mono{
			va.count("mono") ? std::optional(va["mono"].as<value_type>()) : std::nullopt};
//! [Parse command line options]

//! [Prepare spectral and aperture filters]
		const auto [lambda, spectral_filter] = make_spectral_filter(response_filename, mono, carrier);
		const auto aperture_filter = make_aperture_filter(aperture_scale, central_obscuration, square);
//! [Prepare spectral and aperture filters]

		const auto t1 = std::chrono::high_resolution_clock::now();

//! [Create weight function]
		constexpr auto wf_grid_size = 1024 + 1;
		const auto wf = std::visit([&] (const auto& af) {
			return std::visit([&] (const auto& sf) {
				return weif::weight_function<value_type>{sf, lambda, af, aperture_scale, wf_grid_size};
			}, spectral_filter);
		}, aperture_filter);
//! [Create weight function]

		const auto t2 = std::chrono::high_resolution_clock::now();

//! [Evaluate weight function]
		const xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(30), size);

		std::ofstream stm(output_filename);
		xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid, wf(grid)))));
//! [Evaluate weight function]

		std::cerr << "Consumed time: " << std::chrono::duration_cast<std::chrono::duration<value_type>>(t2-t1).count() << " sec" << std::endl;

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

/**
 * @page tutorial_aperture_filter Aperture filter examples
 *
 * @tableofcontents
 *
 * @section overview Overview
 *
 * This tutorial demonstrates how to use the `libweif` library to compute
 * aperture filter functions for different aperture geometries. The example
 * program `aperture_filter` generates three common aperture filters: circular,
 * annular, and cross‑annular, and writes their values to CSV files.
 *
 * Aperture filters describe how the spatial frequency response of an optical
 * system is modified by spatial averaging due to the finite size and shape of
 * the aperture.  They are essential components in the calculation of weight
 * functions for atmospheric turbulence.
 *
 * The `weif::af` namespace provides several aperture filter classes for common
 * aperture geometries.
 *
 * For more details, see the complete source code: \ref aperture_filter.cpp
 *
 * @section numeric_type Choosing the numeric type
 *
 * The example uses `float` as the numeric type for calculations. You can change
 * it to `double` for higher precision if needed. The type alias `value_type` is
 * defined at the beginning of the program.
 *
 * \snippet{lineno,trimleft} aperture_filter.cpp Define value_type
 *
 * @section command_line Command‑line interface
 *
 * The program accepts a single optional command‑line argument that controls the
 * output resolution. The option is defined using Boost.Program_options.  Using
 * Boost.Program_options is generally not required; this part is specific to
 * this particular example program.
 *
 * \snippet{lineno,trimleft} aperture_filter.cpp Define command line options
 *
 * We obtain the following command line interface:
 * \code{bash}
 * > aperture_filter --help
 * --help                Print help message
 * --size arg (=1024)    Output grid size
 * \endcode
 *
 * @section parsing Parsing command‑line options
 *
 * The program uses Boost.Program_options to parse the command line. The parsed
 * values are stored in a `variables_map` and later extracted into appropriate
 * variables.
 *
 * \snippet{lineno,trimleft} aperture_filter.cpp Parse command line options
 *
 * @section aperture_filters Available aperture filters
 *
 * The example evaluates three different aperture filters:
 *
 * 1. **Circular aperture** – a uniform circular aperture of unit radius.
 *    The filter function is the squared jinc function:
 *    \f[
 *    A(u) = \mathrm{jinc}_1^2(\pi u),
 *    \f]
 *    where \f$\mathrm{jinc}_1(x) = 2 J_1(x) / x\f$ and \f$J_1\f$ is the Bessel
 *    function of the first kind.
 *
 * 2. **Annular aperture** – a ring‑shaped aperture with inner radius
 *    \f$\varepsilon = 0.25\f$ and outer radius 1.
 *
 * 3. **Cross‑annular aperture** – an aperture filter for the covariance between
 *    two annular apertures.  The example uses parameters corresponding to the
 *    scintillation covariance between MASS apertures A and B, i.e., between an
 *    annular aperture and its inner disk.
 *
 * @section evaluation Evaluating the filters
 *
 * The program creates an instance of each aperture filter and evaluates it on a
 * uniform grid of dimensionless spatial frequencies. The grid spans from 0 to
 * 5, which covers the typical range where the filter has significant magnitude.
 *
 * \snippet{lineno,trimleft} aperture_filter.cpp Create and evaluate filters
 *
 * The helper function `dump_aperture_filter` writes the grid coordinate and the
 * corresponding filter value to a CSV file.
 *
 * \snippet{lineno,trimleft} aperture_filter.cpp Dump aperture filter
 *
 * @section output Generated output files
 *
 * After a successful run, three CSV files are created in the current directory:
 * - `circular_aperture.csv`
 * - `annular_aperture.csv`
 * - `cross_annular_aperture.csv`
 *
 * Each file contains two columns:
 * 1. The dimensionless spatial frequency \f$u\f$
 * 2. The aperture filter value \f$A(u)\f$
 *
 * If you want to plot the output in dimensional units, use \f$ u = D \cdot f
 * \f$ as the argument, where \f$ D \f$ is the aperture size in meters and \f$ f
 * \f$ is the spatial frequency in \f$ \mathrm{m}^{-1} \f$.
 */

/**
 * @example aperture_filter.cpp
 *
 * This is an example of how to use the aperture filter classes in the
 * \ref weif::af namespace.
 */

#include <iostream>
#include <fstream>
#include <string>

#include <boost/program_options.hpp> // IWYU pragma: keep

#include <xtensor/containers/xarray.hpp> // IWYU pragma: keep
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/io/xcsv.hpp>
#include <xtensor/misc/xmanipulation.hpp>

#include <weif/af/circular.h>

//! [Define value_type]
using value_type = float;
//! [Define value_type]

//! [Dump aperture filter]
template<class AF>
void dump_aperture_filter(const std::string& filename, const AF& af, std::size_t size) {
	using value_type = typename AF::value_type;

	xt::xarray<value_type> grid = xt::linspace(static_cast<value_type>(0), static_cast<value_type>(5), size);

	std::ofstream stm(filename);
	xt::dump_csv(stm, xt::transpose(xt::vstack(xt::xtuple(grid, af(grid)))));
}
//! [Dump aperture filter]


int main(int argc, char** argv) {
	namespace po = boost::program_options;

//! [Define command line options]
	po::options_description opts;
	po::positional_options_description pos_opts;
	po::variables_map va;

	opts.add_options()
		("help", "Print help message")
		("size", po::value<std::size_t>()->default_value(1024), "Output grid size");
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

		const auto size = va["size"].as<std::size_t>();
//! [Parse command line options]
//! [Create and evaluate filters]
		dump_aperture_filter("circular_aperture.csv", weif::af::circular<float>{}, size);
		dump_aperture_filter("annular_aperture.csv", weif::af::annular<float>{0.25}, size);
		dump_aperture_filter("cross_annular_aperture.csv", weif::af::cross_annular<float>{0.5905, 0.6046, 0.0}, size);
//! [Create and evaluate filters]

	} catch (const po::error& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << opts << std::endl;

		return 1;
	}

	return 0;
}

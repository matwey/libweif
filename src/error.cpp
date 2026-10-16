/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <format>

#include <weif/error.h>


namespace weif {

error::error(const std::string& what): std::runtime_error(what) {}
error::error(const char* what): std::runtime_error(what) {}

mismatched_grids::mismatched_grids() noexcept:
	error("Mismatched grids") {}

empty_grid_data::empty_grid_data() noexcept:
	error("Empty grid and data") {}

mismatched_grid_data::mismatched_grid_data(std::size_t grid_size, std::size_t data_size):
	error(std::format("Mismatched grid and data sizes: grid size {}, data size {}", grid_size, data_size)) {}

} // weif

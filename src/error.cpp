/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <weif/error.h>


namespace weif {

error::error(const std::string& what): std::runtime_error(what) {}
error::error(const char* what): std::runtime_error(what) {}

mismatched_grids::mismatched_grids() noexcept:
	error("Mismatched grids") {}

} // weif

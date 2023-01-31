/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <weif/weight_function_2d.h>


namespace weif {

template class weight_function_2d<float>;
template class weight_function_2d<double>;
template class weight_function_2d<long double>;

} // weif

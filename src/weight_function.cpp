/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <weif/weight_function.h>


namespace weif {

template class weight_function<float>;
template class weight_function<double>;
template class weight_function<long double>;

} // weif

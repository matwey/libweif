/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <weif/spectral_filter.h>


namespace weif {

template class spectral_filter<float>;
template class spectral_filter<double>;
template class spectral_filter<long double>;

} // weif

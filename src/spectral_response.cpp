/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <weif/spectral_response.h>


namespace weif {

template class spectral_response<float>;
template class spectral_response<double>;
template class spectral_response<long double>;

} // weif

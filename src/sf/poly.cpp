/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <weif/sf/poly.h>


namespace weif {
namespace sf {

template class WEIF_INSTANTIATION_EXPORT poly<float>;
template class WEIF_INSTANTIATION_EXPORT poly<double>;
template class WEIF_INSTANTIATION_EXPORT poly<long double>;

} // sf
} // weif

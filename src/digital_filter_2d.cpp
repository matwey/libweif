/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <weif/digital_filter_2d.h>


namespace weif {

template class digital_filter_2d<float>;
template class digital_filter_2d<double>;
template class digital_filter_2d<long double>;

#if __cpp_lib_memory_resource >= 201603

template class digital_filter_2d<float, std::pmr::polymorphic_allocator<float>>;
template class digital_filter_2d<double, std::pmr::polymorphic_allocator<double>>;
template class digital_filter_2d<long double, std::pmr::polymorphic_allocator<long double>>;

#endif

} // weif

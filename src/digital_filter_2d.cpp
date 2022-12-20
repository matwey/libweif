#include <digital_filter_2d.h>


namespace weif {

template class digital_filter_2d<float>;
template class digital_filter_2d<double>;
template class digital_filter_2d<long double>;

template class digital_filter_2d<float, std::pmr::polymorphic_allocator<float>>;
template class digital_filter_2d<double, std::pmr::polymorphic_allocator<double>>;
template class digital_filter_2d<long double, std::pmr::polymorphic_allocator<long double>>;

} // weif

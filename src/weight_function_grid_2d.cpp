#include <weight_function_grid_2d.h>


namespace weif {

template class weight_function_grid_2d<float>;
template class weight_function_grid_2d<double>;
template class weight_function_grid_2d<long double>;

template class weight_function_grid_2d<float, std::pmr::polymorphic_allocator<float>>;
template class weight_function_grid_2d<double, std::pmr::polymorphic_allocator<double>>;
template class weight_function_grid_2d<long double, std::pmr::polymorphic_allocator<long double>>;

} // weif

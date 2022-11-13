#ifndef _WEIF_UNIFORM_GRID_H
#define _WEIF_UNIFORM_GRID_H

#include <error.h>

#include <xtensor/xgenerator.hpp>

namespace weif {

namespace detail {

template<class T>
struct uniform_grid_fn {
	using value_type = T;

	value_type origin;
	value_type delta;

	value_type operator() (std::size_t ind) const {
		return origin + ind * delta;
	}

	template<class Iter>
	value_type element(Iter it, Iter) const {
		return operator() (*it);
	}

	template<class Iter,
		std::enable_if_t<
			std::is_convertible<
				typename std::iterator_traits<Iter>::value_type, T>::value, bool> = true>
	static uniform_grid_fn<T> make_from_iterable(Iter begin, Iter end) {
		uniform_grid_fn<T> grid_fn{0, 1};

		if (begin == end)
			return grid_fn;

		grid_fn.origin = *begin++;

		if (begin == end)
			return grid_fn;

		grid_fn.delta = *begin++ - grid_fn.origin;

		for (std::size_t i = 2; begin != end; ++begin, ++i) {
			if (grid_fn(i) != *begin)
				throw non_uniform_grid{i, *begin, grid_fn(i)};
		}

		return grid_fn;
	}
};

} // detail

template<class T>
class uniform_grid:
	public xt::xgenerator<detail::uniform_grid_fn<T>, T, std::array<std::size_t, 1>> {
public:
	using value_type = T;

	uniform_grid(value_type origin, value_type delta, std::size_t size) noexcept:
		xt::xgenerator<detail::uniform_grid_fn<T>, T, std::array<std::size_t, 1>>{
			detail::uniform_grid_fn<T>{origin, delta},
			{size}} {}

	template<class Iter>
	uniform_grid(Iter begin, Iter end):
		xt::xgenerator<detail::uniform_grid_fn<T>, T, std::array<std::size_t, 1>>{
			detail::uniform_grid_fn<T>::make_from_iterable(begin, end),
			{std::distance(begin, end)}} {}

	const value_type& origin() const noexcept {
		return this->functor().origin;
	}
	const value_type& delta() const noexcept {
		return this->functor().delta;
	}
};

template<class Iter>
uniform_grid(Iter begin, Iter end) -> uniform_grid<typename std::iterator_traits<Iter>::value_type>;

} // weif

#endif // _WEIF_UNIFORM_GRID_H

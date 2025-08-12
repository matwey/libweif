/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2022-2023  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#ifndef _WEIF_UNIFORM_GRID_H
#define _WEIF_UNIFORM_GRID_H

#include <cstdlib>

#include <weif/detail/uniform_grid.h>
#include <weif_export.h>

#include <xtensor/xgenerator.hpp>

namespace weif {

/**
 * @brief Uniformly spaced numerical grid
 *
 * @tparam T Numeric type for grid values
 *
 * Represents a 1D grid of uniformly spaced values, providing mathematical
 * operations and grid analysis capabilities. The grid is defined by:
 * \f[
 * x_i = x_0 + i \cdot \Delta x \quad \text{for} \quad i = 0,1,...,N-1
 * \f]
 * where x_0 is the grid origin, \f$\Delta x\f$ is the grid spacing, and \f$N\f$ is the grid size.
 */
template<class T>
class WEIF_EXPORT uniform_grid {
public:
	using value_type = T; ///< Numeric type for grid values

private:
	using generator_type = xt::xgenerator<detail::uniform_grid_fn<T>, T, std::array<std::size_t, 1>>;

	generator_type generator_;

	const detail::uniform_grid_fn<T>& functor() const noexcept {
		return generator_.functor();
	}
public:

	/**
	 * @brief Construct from origin, spacing and size
	 * @param origin Starting value of the grid
	 * @param delta Spacing between grid points
	 * @param size Number of grid points
	 */
	uniform_grid(value_type origin, value_type delta, std::size_t size) noexcept:
		generator_{detail::uniform_grid_fn<T>{origin, delta}, {size}} {}

	/**
	 * @brief Construct from iterator range
	 * @tparam Iter Iterator type
	 * @param begin Start iterator
	 * @param end End iterator
	 *
	 * This constructor is particularly useful for validating input data.
	 *
	 * @throws non_uniform_grid If the iterator range does not define a uniform grid
	 *
	 * Computes uniform grid parameters from the input range.
	 * Requires at least 2 elements to determine the spacing.
	 */
	template<class Iter>
	uniform_grid(Iter begin, Iter end):
		generator_{detail::uniform_grid_fn<T>::make_from_iterable(begin, end),
			{static_cast<std::size_t>(std::distance(begin, end))}} {}

	/// @return True if grids are identical
	bool operator==(const uniform_grid<T>& other) const noexcept {
		return generator_ == other.generator_;
	}

	/// @return True if grids differ
	bool operator!=(const uniform_grid<T>& other) const noexcept {
		return !(*this == other);
	}

	/// @return The starting value of the grid
	const value_type& origin() const noexcept {
		return functor().origin;
	}

	/// @return The spacing between grid points
	const value_type& delta() const noexcept {
		return functor().delta;
	}

	/// @return The grid values as an xtensor generator expression
	const generator_type& values() const noexcept {
		return generator_;
	}

	/// @return Number of points in the grid
	std::size_t size() const noexcept { return generator_.size(); }

	/**
	 * @brief Check if grids are compatible, i.e have same spacing and phase
	 * @param other Grid to compare with
	 * @return True if grids can be intersected
	 */
	bool match(const uniform_grid<T>& other) const noexcept {
		using namespace std;

		return delta() == other.delta()
			&& fmod(origin(), delta()) == fmod(other.origin(), other.delta());
	}

	/**
	 * @brief Computes the intersection of two uniform grids
	 *
	 * @param other Grid to intersect with
	 * @return New grid representing the overlapping range
	 *
	 * @par The intersection operation:
	 * - Finds the common range where both grids overlap
	 * - Creates a new grid with the same spacing as the input grid
	 * - Preserves only the overlapping portion of the grids
	 *
	 * @see match()
	 * @throws mismatched_grids If the grids can not be intersected
	 */
	uniform_grid<T> intersect(const uniform_grid<T>& other) const {
		if (other.origin() < origin())
			return other.intersect(*this);

		if (!match(other))
			throw mismatched_grids{};

		const auto new_size = (size() == 0 || other.size() == 0 || *values().crbegin() < other.origin() ?
			static_cast<std::size_t>(0) :
			static_cast<std::size_t>((std::min(*values().crbegin(), *other.values().crbegin()) - other.origin()) / other.delta()) + 1);

		return {other.origin(), other.delta(), new_size};
	}

	/**
	 * @brief Find index of nearest grid point
	 * @param v Value to locate
	 * @return Index of the grid point
	 */
	std::size_t to_index(value_type v) const noexcept {
		return static_cast<std::size_t>((v - origin()) / delta());
	}

	template<class U>
	auto operator+ (const U x) const noexcept {
		return uniform_grid<T>{origin() + x, delta(), size()};
	}
	template<class U>
	auto operator- (const U x) const noexcept {
		return uniform_grid<T>{origin() - x, delta(), size()};
	}
	template<class U>
	auto operator* (const U x) const noexcept {
		return uniform_grid<T>{origin() * x, delta() * x, size()};
	}
	template<class U>
	auto operator/ (const U x) const noexcept {
		return uniform_grid<T>{origin() / x, delta() / x, size()};
	}

	template<class U>
	auto operator+= (const U x) noexcept {
		return *this = *this + x;
	}
	template<class U>
	auto operator-= (const U x) noexcept {
		return *this = *this - x;
	}
	template<class U>
	auto operator*= (const U x) noexcept {
		return *this = *this * x;
	}
	template<class U>
	auto operator/= (const U x) noexcept {
		return *this = *this / x;
	}
};

template<class T>
inline bool operator< (const uniform_grid<T>& lhs, const T& rhs) {
	return *lhs.values().crbegin() < rhs;
}

template<class T>
inline bool operator> (const uniform_grid<T>& lhs, const T& rhs) {
	return lhs.origin() > rhs;
}

template<class T>
inline bool operator<=(const uniform_grid<T>& lhs, const T& rhs) {
	return *lhs.values().crbegin() <= rhs;
}

template<class T>
inline bool operator>=(const uniform_grid<T>& lhs, const T& rhs) {
	return lhs.origin() >= rhs;
}

template<class Iter>
uniform_grid(Iter begin, Iter end) -> uniform_grid<typename std::iterator_traits<Iter>::value_type>;

} // weif

#endif // _WEIF_UNIFORM_GRID_H

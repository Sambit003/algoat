/**
 * @file utils.hpp
 * @brief Core utility algorithms and helpers avoiding standard library overhead.
 */

#pragma once

#include <cassert>

namespace algoat {

/**
 * @brief Clamps a value within the inclusive range [lo, hi].
 * @tparam T Type satisfying strict weak ordering.
 * @param v Value to clamp.
 * @param lo Lower bound.
 * @param hi Upper bound.
 * @return Clamped reference to @p v, @p lo, or @p hi.
 */
template <class T> constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
    assert(!(hi < lo) && "clamp requires lo <= hi");
    return (v < lo) ? lo : ((hi < v) ? hi : v);
}

} // namespace algoat

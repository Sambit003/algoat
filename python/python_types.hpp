/**
 * @file python_types.hpp
 * @brief Custom user-defined types exposed to Python test suites and benchmarks.
 */

#pragma once
#include <cstdint>

namespace algoat {

/**
 * @struct Rational
 * @brief Exact fractional number representation (numerator / denominator).
 * 
 * Implements cross-multiplication comparison without floating-point rounding errors.
 * Uses 128-bit integer arithmetic (`__int128`) on supported platforms to prevent intermediate overflow.
 */
struct Rational {
    int64_t num; ///< Numerator.
    int64_t den; ///< Denominator.

    bool operator<(const Rational& other) const {
#if defined(__SIZEOF_INT128__)
        __int128 lhs = static_cast<__int128>(num) * other.den;
        __int128 rhs = static_cast<__int128>(other.num) * den;
        return lhs < rhs;
#else
        return num * other.den < other.num * den;
#endif
    }
    bool operator==(const Rational& other) const {
        return num * other.den == other.num * den;
    }
};

/**
 * @struct Vec3f
 * @brief 3D vector of single-precision floating point components.
 */
struct Vec3f {
    float x, y, z;
};

} // namespace algoat

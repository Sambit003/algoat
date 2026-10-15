/******************************************************************************
 * algoat/sorting/detail/ips4o/utils.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#ifndef IPS4O_ASSUME_NOT
#if defined(__GNUC__) || defined(__clang__)
#define IPS4O_ASSUME_NOT(c)                                                                        \
    if (c)                                                                                         \
    __builtin_unreachable()
#elif defined(_MSC_VER)
#define IPS4O_ASSUME_NOT(c) __assume(!(c))
#else
#define IPS4O_ASSUME_NOT(c)                                                                        \
    do {                                                                                           \
    } while (0)
#endif
#endif

#include <bit>
#include <limits>

namespace algoat::sorting::detail::ips4o {
namespace detail {

/**
 * Compute the logarithm to base 2, rounded down.
 */
inline constexpr unsigned long log2(unsigned long n) {
    return n == 0 ? 0
                  : (static_cast<unsigned long>(std::numeric_limits<unsigned long>::digits - 1 -
                                                std::countl_zero(n)));
}

} // namespace detail
} // namespace algoat::sorting::detail::ips4o

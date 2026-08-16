/**
 * @file traits.hpp
 * @brief Data profiling and traits analysis for runtime algorithm selection.
 * 
 * Provides utilities to inspect input sequences in a single O(n) pass to measure
 * characteristics like size, sortedness ratio, and presence of duplicate elements.
 */

#pragma once

#include <cstddef>
#include <ranges>

namespace algoat::core {

/**
 * @struct DataTraits
 * @brief Represents statistical and structural properties of a dataset.
 * 
 * Used by @c algoat::core::Dispatcher to make intelligent heuristic decisions
 * regarding algorithm selection (e.g., choosing TimSort for nearly sorted arrays
 * or InsertionSort for small arrays).
 */
struct DataTraits {
    std::size_t size;         ///< Number of elements in the range.
    double sortedness_ratio;  ///< Sortedness metric: 0.0 = completely reversed, 1.0 = fully sorted.
    bool has_duplicates;      ///< True if at least one adjacent duplicate pair was detected.
};

/**
 * @brief Analyzes a random access range in a single O(N) pass to extract structural traits.
 * 
 * Scans adjacent pairs to compute the sortedness ratio:
 * @code{.text}
 * ratio = count(data[i-1] <= data[i]) / (N - 1)
 * @endcode
 * - Arrays with <tt>ratio >= 0.90</tt> are considered nearly sorted.
 * - Arrays with <tt>ratio <= 0.10</tt> indicate nearly reverse-sorted data.
 * 
 * @tparam R A type satisfying @c std::ranges::random_access_range.
 *
 * @param data The range of elements to analyze.
 * @return @c DataTraits Computed traits (@c size, @c sortedness_ratio, @c has_duplicates).
 * 
 * @note Time complexity is <tt>O(N)</tt>, performing exactly <tt>(N - 1)</tt> comparisons with <tt>O(1)</tt> auxiliary space.
 */
template <std::ranges::random_access_range R>
DataTraits analyze(const R& data) {
    const std::size_t size = std::ranges::size(data);
    if (size <= 1) {
        return {size, 1.0, false};
    }

    std::size_t sorted_pairs = 0;
    bool has_duplicates = false;

    auto it = std::ranges::begin(data);
    auto prev = it;
    ++it;

    for (; it != std::ranges::end(data); ++it, ++prev) {
        if (*prev <= *it) {
            sorted_pairs++;
        }
        if (*prev == *it) {
            has_duplicates = true;
        }
    }

    // A fully sorted array has (size - 1) sorted pairs.
    double ratio = static_cast<double>(sorted_pairs) / static_cast<double>(size - 1);

    return {size, ratio, has_duplicates};
}

} // namespace algoat::core

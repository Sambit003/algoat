/**
 * @file traits.hpp
 * @brief Data profiling and traits analysis for runtime algorithm selection.
 *
 * Provides utilities to inspect input sequences in a single O(n) pass to measure
 * characteristics like size, sortedness ratio, and presence of duplicate elements.
 */

#pragma once

#include "algoat/core/simd_profiler.hpp"

#include <cstddef>
#include <ranges>
#include <type_traits>

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
    std::size_t size;        ///< Number of elements in the range.
    double sortedness_ratio; ///< Sortedness metric: 0.0 = completely reversed, 1.0 = fully sorted.
    bool has_duplicates;     ///< True if at least one adjacent duplicate pair was detected.
    bool is_exact{
        true}; ///< True if traits were calculated via exact scan; false if sampled via SIMD.
};

/**
 * @brief Analyzes a random access range to extract structural traits.
 *
 * For small sequences (N <= 10,000) or non-contiguous ranges, performs an exact O(N) pass.
 * For large contiguous sequences (N > 10,000), executes a sub-linear O(1) cache-line-aware
 * stratified SIMD profiler with high statistical confidence.
 *
 * @tparam R A type satisfying @c std::ranges::random_access_range.
 *
 * @param data The range of elements to analyze.
 * @return @c DataTraits Computed traits (@c size, @c sortedness_ratio, @c has_duplicates, @c
 * is_exact).
 */
template <std::ranges::random_access_range R> DataTraits analyze(const R& data) {
    const std::size_t size = std::ranges::size(data);
    if (size <= 1) {
        return {size, 1.0, false, true};
    }

    using ElementType = std::remove_cvref_t<std::ranges::range_value_t<R>>;

    // Sub-linear fast path for large contiguous primitive sequences
    if constexpr (std::ranges::contiguous_range<R> &&
                  (std::is_arithmetic_v<ElementType> || std::is_pointer_v<ElementType>)) {
        if (size > detail::kSublinearThreshold) {
            double ratio = 1.0;
            bool has_duplicates = false;
            detail::sample_traits_sublinear_impl(std::ranges::data(data), size, ratio,
                                                 has_duplicates);
            return {size, ratio, has_duplicates, /*is_exact=*/false};
        }
    }

    // Exact O(N) scalar pass for small arrays or non-contiguous/custom ranges
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

    return {size, ratio, has_duplicates, /*is_exact=*/true};
}

} // namespace algoat::core

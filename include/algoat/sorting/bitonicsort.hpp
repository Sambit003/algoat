/**
 * @file bitonicsort.hpp
 * @brief Bitonic Sort sorting network implementation.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <algorithm>
#include <cassert>
#include <bit>
#include <stdexcept>

namespace algoat::sorting {

/**
 * @struct BitonicSort
 * @brief Parallel sorting network algorithm designed for power-of-two dataset sizes.
 * 
 * Recursively creates bitonic sequences (sequences that first monotonically increase,
 * then monotonically decrease) and merges them. Requires the input span size to be a
 * power of 2 (<tt>N = 2^k</tt>).
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Sorting Network.
 * - <b>Stability:</b> Unstable.
 * - <b>Constraint:</b> Input size must satisfy <tt>std::has_single_bit(N)</tt>.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log^2 N)
 * - Average Case: @c O(N log^2 N)
 * - Worst Case: @c O(N log^2 N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(log^2 N) recursion stack space
 */
struct BitonicSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "bitonicsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "bitonicsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Bitonic merge worker recursively sorting bitonic subranges.
 *
 * @param a Span to merge.
 *
 * @param dir Sort direction (@c true for ascending, @c false for descending).
     */
    template<std::totally_ordered T>
    static void bitonic_merge(std::span<T> a, bool dir) {
        if (a.size() > 1) {
            std::size_t k = a.size() / 2;
            for (std::size_t i = 0; i < k; ++i) {
                if (dir == (a[i] > a[i + k])) {
                    std::swap(a[i], a[i + k]);
                }
            }
            bitonic_merge(a.subspan(0, k), dir);
            bitonic_merge(a.subspan(k, k), dir);
        }
    }

    /**
     * @brief Recursive worker building ascending and descending halves before bitonic merge.
     */
    template<std::totally_ordered T>
    static void bitonic_sort_impl(std::span<T> a, bool dir) {
        if (a.size() > 1) {
            std::size_t k = a.size() / 2;
            bitonic_sort_impl(a.subspan(0, k), true);
            bitonic_sort_impl(a.subspan(k, k), false);
            bitonic_merge(a, dir);
        }
    }

    /**
     * @brief Sorts the span in-place using bitonic sorting network.
     * @tparam T Type satisfying @c std::totally_ordered.
 *
 * @param data Span of elements to sort (must be power of 2 size).
     * @throws std::invalid_argument If <tt>data.size()</tt> is not a power of 2.
     */
    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        if (data.empty()) return;
        if (!std::has_single_bit(data.size())) {
            throw std::invalid_argument("Bitonic sort requires array size to be a power of 2");
        }
        bitonic_sort_impl(data, true);
    }
};

} // namespace algoat::sorting

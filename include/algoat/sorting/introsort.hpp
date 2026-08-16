/**
 * @file introsort.hpp
 * @brief Introspective Sort (IntroSort) hybrid algorithm implementation.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <algorithm>
#include <bit>
#include "algoat/sorting/insertionsort.hpp"
#include "algoat/sorting/heapsort.hpp"

namespace algoat::sorting {

/**
 * @struct IntroSort
 * @brief Hybrid sorting algorithm combining QuickSort, HeapSort, and InsertionSort.
 * 
 * Developed by David Musser (1997), IntroSort begins with QuickSort and monitors
 * recursion depth. If the recursion depth exceeds <tt>2 * floor(log2(N))</tt>, it
 * switches to HeapSort to prevent quadratic worst-case degradation. Small partitions
 * (<tt><= 32</tt> elements) are sorted using InsertionSort.
 *
 * @par Characteristics:
 * - <b>Category:</b> Hybrid (QuickSort + HeapSort + InsertionSort).
 * - <b>Stability:</b> Unstable
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N log N) (guaranteed by HeapSort fallback)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(log N) recursion stack space
 */
struct IntroSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "introsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "introsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Computes minimum of two totally ordered values by value.
     */
    template<std::totally_ordered T>
    static T min_val(const T& a, const T& b) {
        return (b < a) ? b : a;
    }

    /**
     * @brief Computes maximum of two totally ordered values by value.
     */
    template<std::totally_ordered T>
    static T max_val(const T& a, const T& b) {
        return (a < b) ? b : a;
    }

    /**
     * @brief Median-of-three pivot selection avoiding reference-aliasing optimizer bottlenecks.
     * 
     * Uses custom @c min_val/@c max_val returning by VALUE rather than reference.
     * This avoids nested reference chains that prevent compilers from generating
     * branchless register CMOV instructions.
     */
    template<std::totally_ordered T>
    static T get_pivot(const T& a, const T& b, const T& c) {
        return max_val(min_val(a, b), min_val(max_val(a, b), c));
    }

    /**
     * @brief Recursive implementation of introsort with depth limiting and Hoare partitioning.
 *
 * @param data Subspan to sort.
 *
 * @param depth_limit Remaining recursion budget before switching to HeapSort.
     */
    template<std::totally_ordered T>
    static void introsort_impl(std::span<T> data, int depth_limit) {
        if (data.size() <= 32) {
            InsertionSort{}.sort(data);
            return;
        }

        if (depth_limit == 0) {
            HeapSort{}.sort(data);
            return;
        }

        // Median-of-three pivot selection
        auto pivot = get_pivot(data[0], data[data.size() / 2], data[data.size() - 1]);
        
        // Hoare partition scheme
        auto* left = data.data() - 1;
        auto* right = data.data() + data.size();
        
        while (true) {
            do { left++; } while (*left < pivot);
            do { right--; } while (pivot < *right);
            if (left >= right) break;
            std::swap(*left, *right);
        }
        
        std::size_t pivot_idx = right - data.data() + 1;

        introsort_impl(data.subspan(0, pivot_idx), depth_limit - 1);
        introsort_impl(data.subspan(pivot_idx, data.size() - pivot_idx), depth_limit - 1);
    }

    /**
     * @brief Sorts the span in-place using introsort.
     * @tparam T Type satisfying @c std::totally_ordered.
 *
 * @param data Contiguous span of elements to sort.
     */
    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        if (data.empty()) return;
        int depth_limit = 2 * std::bit_width(data.size());
        introsort_impl(data, depth_limit);
    }
};

} // namespace algoat::sorting

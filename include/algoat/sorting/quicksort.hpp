/**
 * @file quicksort.hpp
 * @brief Quick Sort implementation with Dijkstra 3-Way Fat Partitioning (Dutch National Flag).
 */

#pragma once

#include "algoat/sorting/insertionsort.hpp"

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

/**
 * @struct QuickSort
 * @brief Divide-and-conquer sorting algorithm using Dijkstra 3-Way Fat Partitioning (Dutch National
 * Flag).
 *
 * Partitions array ranges into three contiguous slices: <tt>[ < pivot | == pivot | > pivot ]</tt>.
 * Duplicate keys equal to the pivot are finalized in place during the current partitioning pass,
 * reducing duplicate-heavy and low-entropy inputs to optimal @c O(N) linear time and preventing
 * Lomuto quadratic degradation. Employs median-of-three pivot selection and tail-call recursion
 * depth bounding to guarantee @c O(log N) stack depth.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Divide & Conquer (3-Way Partitioning).
 * - <b>Stability:</b> Unstable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N) (all duplicate keys / low entropy) or @c O(N log N)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N^2) (mitigated by median-of-three and 3-way partitioning; see IntroSort for
 * guaranteed O(N log N))
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(log N) recursion stack space (guaranteed by tail-call elimination)
 */
struct QuickSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "quicksort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "quicksort";
    }

    /**
     * @brief Sorts the span in-place using 3-way partitioning quicksort.
     * @tparam T Element type supporting <tt>operator<</tt> and <tt>operator></tt>.
     *
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> void sort(std::span<T> data) const {
        if (data.size() <= 1)
            return;
        quicksort_impl(data.data(), 0, data.size() - 1);
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 32 (arrays smaller than 32 elements are better sorted via InsertionSort).
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 32;
    }

private:
    /**
     * @brief Result structure holding the boundary indices of equal-to-pivot partition.
     */
    struct PartitionResult {
        std::size_t lt;
        std::size_t gt;
    };

    /**
     * @brief Selects the median of <tt>arr[low]</tt>, <tt>arr[mid]</tt>, and <tt>arr[high]</tt>
     * and places it at <tt>arr[high]</tt>.
     */
    template <typename T> void median_of_three(T* arr, std::size_t low, std::size_t high) const {
        std::size_t mid = low + (high - low) / 2;
        if (arr[mid] < arr[low])
            std::swap(arr[low], arr[mid]);
        if (arr[high] < arr[low])
            std::swap(arr[low], arr[high]);
        if (arr[high] < arr[mid])
            std::swap(arr[mid], arr[high]);
        // Place the median at arr[high] so it is excluded from the active partition scan [low, high
        // - 1]
        std::swap(arr[mid], arr[high]);
    }

    /**
     * @brief Dijkstra 3-Way Fat Partitioning (Dutch National Flag) with zero-copy pivot reference.
     *
     * Stashes the pivot at <tt>arr[high]</tt> and partitions <tt>arr[low..high-1]</tt> against
     * <tt>const T& pivot = arr[high]</tt> into:
     * - <tt>arr[low..lt-1] < pivot</tt>
     * - <tt>arr[lt..gt] == pivot</tt>
     * - <tt>arr[gt+1..high-1] > pivot</tt>
     *
     * Finally, moves the pivot from <tt>arr[high]</tt> to <tt>arr[gt+1]</tt> and expands @c gt.
     *
     * @return Boundary indices <tt>{lt, gt}</tt>.
     */
    template <typename T>
    PartitionResult partition_3way(T* arr, std::size_t low, std::size_t high) const {
        median_of_three(arr, low, high);
        const T& pivot = arr[high];
        std::size_t lt = low;
        std::size_t i = low;
        std::size_t gt = high - 1;

        while (i <= gt) {
            if (arr[i] < pivot) {
                std::swap(arr[lt], arr[i]);
                ++lt;
                ++i;
            } else if (pivot < arr[i]) {
                std::swap(arr[i], arr[gt]);
                if (gt == 0)
                    break;
                --gt;
            } else {
                ++i;
            }
        }

        std::swap(arr[gt + 1], arr[high]);
        ++gt;
        return {lt, gt};
    }

    /**
     * @brief Iterative/recursive quicksort with tail-call elimination.
     *
     * Delegates small partitions directly to InsertionSort, recurses on the smaller partition,
     * and iterates on the larger partition to strictly guarantee @c O(log N) maximum stack depth.
     */
    template <typename T> void quicksort_impl(T* arr, std::size_t low, std::size_t high) const {
        while (low < high) {
            if (high - low + 1 <= 16) {
                InsertionSort{}.sort(std::span<T>{arr + low, high - low + 1});
                return;
            }

            auto [lt, gt] = partition_3way(arr, low, high);

            std::size_t left_size = (lt > low) ? (lt - low) : 0;
            std::size_t right_size = (high > gt) ? (high - gt) : 0;

            if (left_size < right_size) {
                if (left_size > 0 && lt > 0) {
                    quicksort_impl(arr, low, lt - 1);
                }
                if (right_size == 0) {
                    break;
                }
                low = gt + 1;
            } else {
                if (right_size > 0) {
                    quicksort_impl(arr, gt + 1, high);
                }
                if (left_size == 0 || lt == 0) {
                    break;
                }
                high = lt - 1;
            }
        }
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

inline const auto registrar_quicksort = []() {
    ::algoat::core::BaseRegistry::global("sorting").register_algorithm(
        "quicksort", []() -> std::any { return ::algoat::sorting::QuickSort{}; });
    return 0;
}();

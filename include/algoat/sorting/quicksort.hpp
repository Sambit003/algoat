/**
 * @file quicksort.hpp
 * @brief Quick Sort implementation with Dijkstra 3-Way Fat Partitioning (Dutch National Flag).
 */

#pragma once

#include "algoat/simd/partition.hpp"
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
        // Threshold where SIMD vectorization overhead is overcome by parallel throughput
        constexpr std::size_t SIMD_THRESHOLD = 128;

        auto recurse_tail_call = [&](std::size_t left_end, std::size_t right_start) {
            std::size_t left_size = (left_end > low) ? (left_end - low) : 0;
            std::size_t right_size = (high > right_start) ? (high - right_start) : 0;

            if (left_size < right_size) {
                if (left_size > 0 && left_end > 0) {
                    quicksort_impl(arr, low, left_end - 1);
                }
                if (right_size == 0) {
                    return false; // Break loop
                }
                low = right_start + 1;
            } else {
                if (right_size > 0) {
                    quicksort_impl(arr, right_start + 1, high);
                }
                if (left_size == 0 || left_end == 0) {
                    return false; // Break loop
                }
                high = left_end - 1;
            }
            return true; // Continue loop
        };

        while (low < high) {
            std::size_t current_size = high - low + 1;
            if (current_size <= 16) {
                InsertionSort{}.sort(std::span<T>{arr + low, current_size});
                return;
            }

            if constexpr (::algoat::simd::is_simd_supported_v<T>) {
                if (current_size > SIMD_THRESHOLD) {
                    median_of_three(arr, low, high);
                    const T& pivot = arr[high];

                    // Proceed with SIMD partitioning only if the array is not an extreme duplicate
                    // sequence
                    if (arr[low] != pivot) {
                        std::size_t lt_count =
                            ::algoat::simd::partition_simd(arr + low, current_size - 1, pivot);
                        std::size_t split_idx = low + lt_count;

                        std::swap(arr[split_idx], arr[high]);

                        if (!recurse_tail_call(split_idx, split_idx))
                            break;
                        continue;
                    }
                }
            }

            auto [lt, gt] = partition_3way(arr, low, high);
            if (!recurse_tail_call(lt, gt))
                break;
        }
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "quicksort", ::algoat::sorting::QuickSort)

/**
 * @file quicksort.hpp
 * @brief Quick Sort implementation with median-of-three pivot selection.
 */

#pragma once

#include <string_view>
#include <span>
#include <cstddef>
#include <utility>

namespace algoat::sorting {

/**
 * @struct QuickSort
 * @brief Divide-and-conquer sorting algorithm using median-of-three Lomuto partitioning.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Divide & Conquer.
 * - <b>Stability:</b> Unstable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N^2) (mitigated by median-of-three)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(log N) recursion stack space
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
     * @brief Sorts the span in-place using quicksort.
     * @tparam T Element type supporting <tt>operator<</tt> and <tt>operator<=</tt>.
 *
 * @param data Contiguous span of elements to sort.
     */
    template<typename T>
    void sort(std::span<T> data) const {
        if (data.size() <= 1) return;
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
     * @brief Recursive quicksort helper.
 *
 * @param arr Pointer to the raw buffer.
 *
 * @param low Starting index of subrange.
 *
 * @param high Ending index of subrange (inclusive).
     */
    template<typename T>
    void quicksort_impl(T* arr, std::size_t low, std::size_t high) const {
        if (low < high) {
            std::size_t pi = partition(arr, low, high);
            if (pi > 0) {
                quicksort_impl(arr, low, pi - 1);
            }
            quicksort_impl(arr, pi + 1, high);
        }
    }

    /**
     * @brief Sorts <tt>arr[low]</tt>, <tt>arr[mid]</tt>, and <tt>arr[high]</tt> to choose the median as pivot.
     * @return Index of the pivot element (placed at @c high).
     */
    template<typename T>
    std::size_t median_of_three(T* arr, std::size_t low, std::size_t high) const {
        std::size_t mid = low + (high - low) / 2;
        if (arr[mid] < arr[low]) std::swap(arr[low], arr[mid]);
        if (arr[high] < arr[low]) std::swap(arr[low], arr[high]);
        if (arr[mid] < arr[high]) std::swap(arr[mid], arr[high]);
        return high; // pivot is now at high
    }

    /**
     * @brief Partitions the subrange <tt>[low, high]</tt> around the median-of-three pivot.
     * @return Final index position of the pivot.
     */
    template<typename T>
    std::size_t partition(T* arr, std::size_t low, std::size_t high) const {
        median_of_three(arr, low, high);
        const T& pivot = arr[high];
        std::size_t i = low;

        for (std::size_t j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                std::swap(arr[i], arr[j]);
                i++;
            }
        }
        std::swap(arr[i], arr[high]);
        return i;
    }
};

} // namespace algoat::sorting

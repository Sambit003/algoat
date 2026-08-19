/**
 * @file quicksort.hpp
 * @brief Quick Sort implementation with median-of-three pivot and 3-way partitioning.
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

/**
 * @struct QuickSort
 * @brief Divide-and-conquer sorting algorithm using median-of-three pivot selection
 *        and Dijkstra/Bentley-McIlroy 3-way partitioning.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Divide & Conquer.
 * - <b>Stability:</b> Unstable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N log N) (3-way partitioning handles duplicates efficiently)
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
     * @tparam T Element type supporting <tt>operator<</tt> and <tt>operator==</tt>.
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
     * @brief Recursive quicksort helper with 3-way partitioning.
     *
     * @param arr Pointer to the raw buffer.
     * @param low Starting index of subrange.
     * @param high Ending index of subrange (inclusive).
     */
    template <typename T> void quicksort_impl(T* arr, std::size_t low, std::size_t high) const {
        while (low < high) {
            // Use insertion sort for small subarrays
            if (high - low < 32) {
                insertion_sort(arr, low, high);
                return;
            }

            // Median-of-three pivot selection
            std::size_t mid = low + (high - low) / 2;
            if (arr[mid] < arr[low])
                std::swap(arr[low], arr[mid]);
            if (arr[high] < arr[low])
                std::swap(arr[low], arr[high]);
            if (arr[mid] < arr[high])
                std::swap(arr[mid], arr[high]);

            const T& pivot = arr[high];

            // Dijkstra/Bentley-McIlroy 3-way partitioning
            // All elements < pivot go to [low, lt-1]
            // All elements == pivot go to [lt, gt]
            // All elements > pivot go to [gt+1, high]
            std::size_t lt = low;   // less-than pointer
            std::size_t i = low;    // current element
            std::size_t gt = high;  // greater-than pointer

            while (i <= gt) {
                if (arr[i] < pivot) {
                    std::swap(arr[lt], arr[i]);
                    ++lt;
                    ++i;
                } else if (pivot < arr[i]) {
                    std::swap(arr[i], arr[gt]);
                    --gt;
                } else {
                    ++i;
                }
            }

            // Recurse into smaller partition, loop for larger (tail-call optimization)
            if (lt - low < high - gt) {
                quicksort_impl(arr, low, lt - 1);
                low = gt + 1;
            } else {
                quicksort_impl(arr, gt + 1, high);
                high = lt - 1;
            }
        }
    }

    /**
     * @brief Insertion sort for small subarrays.
     */
    template <typename T> void insertion_sort(T* arr, std::size_t low, std::size_t high) const {
        for (std::size_t i = low + 1; i <= high; ++i) {
            T key = std::move(arr[i]);
            std::size_t j = i;
            while (j > low && key < arr[j - 1]) {
                arr[j] = std::move(arr[j - 1]);
                --j;
            }
            arr[j] = std::move(key);
        }
    }
};

} // namespace algoat::sorting

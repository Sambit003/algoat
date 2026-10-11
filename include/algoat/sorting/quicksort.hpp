/**
 * @file quicksort.hpp
 * @brief Quick Sort implementation with Dijkstra 3-Way Fat Partitioning (Dutch National Flag).
 */

#pragma once

#include "algoat/core/traits.hpp"
#include "algoat/simd/partition.hpp"
#include "algoat/sorting/insertionsort.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

namespace detail {

/**
 * @brief Result structure holding the boundary indices of equal-to-pivot partition.
 */
struct PartitionResult {
    std::size_t lt;
    std::size_t gt;
};

/**
 * @brief Selects the median of @c arr[low], @c arr[mid], and @c arr[high]
 * and places it at @c arr[high].
 *
 * @tparam I Random-access iterator type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param arr Base random-access iterator.
 * @param low Starting index.
 * @param high Ending index.
 * @param comp Strict weak ordering comparator.
 */
template <std::random_access_iterator I, typename Compare>
constexpr void median_of_three(I arr, std::size_t low, std::size_t high, Compare comp) {
    std::size_t mid = low + (high - low) / 2;
    if (comp(arr[mid], arr[low]))
        std::ranges::iter_swap(arr + low, arr + mid);
    if (comp(arr[high], arr[low]))
        std::ranges::iter_swap(arr + low, arr + high);
    if (comp(arr[high], arr[mid]))
        std::ranges::iter_swap(arr + mid, arr + high);
    // Place the median at arr[high] so it is excluded from the active partition scan [low, high -
    // 1]
    std::ranges::iter_swap(arr + mid, arr + high);
}

/**
 * @brief Dijkstra 3-Way Fat Partitioning (Dutch National Flag) with zero-copy pivot reference.
 *
 * @tparam I Random-access iterator type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param arr Base random-access iterator.
 * @param low Starting index.
 * @param high Ending index.
 * @param comp Strict weak ordering comparator.
 * @return PartitionResult containing lt and gt boundaries.
 */
template <std::random_access_iterator I, typename Compare>
constexpr PartitionResult partition_3way(I arr, std::size_t low, std::size_t high, Compare comp) {
    const auto& pivot = arr[high];
    std::size_t lt = low;
    std::size_t i = low;
    std::size_t gt = high - 1;

    while (i <= gt) {
        if (comp(arr[i], pivot)) {
            std::ranges::iter_swap(arr + lt, arr + i);
            ++lt;
            ++i;
        } else if (comp(pivot, arr[i])) {
            std::ranges::iter_swap(arr + i, arr + gt);
            if (gt == 0)
                break;
            --gt;
        } else {
            ++i;
        }
    }

    std::ranges::iter_swap(arr + gt + 1, arr + high);
    ++gt;
    return {lt, gt};
}

/**
 * @brief Iterative/recursive quicksort with tail-call elimination.
 *
 * @tparam I Random-access iterator type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param arr Base random-access iterator.
 * @param low Starting index.
 * @param high Ending index.
 * @param comp Strict weak ordering comparator.
 */
template <std::random_access_iterator I, typename Compare>
constexpr void quicksort_impl(I arr, std::size_t low, std::size_t high, Compare comp) {
    // Threshold where SIMD vectorization overhead is overcome by parallel throughput
    constexpr std::size_t SIMD_THRESHOLD = 128;

    auto recurse_smaller_and_advance = [&](std::size_t left_end, std::size_t right_start) {
        std::size_t left_size = (left_end > low) ? (left_end - low) : 0;
        std::size_t right_size = (high > right_start) ? (high - right_start) : 0;

        if (left_size < right_size) {
            if (left_size > 0 && left_end > 0) {
                quicksort_impl(arr, low, left_end - 1, comp);
            }
            if (right_size == 0) {
                low = high; // sentinel: terminate loop
                return;
            }
            low = right_start + 1;
        } else {
            if (right_size > 0) {
                quicksort_impl(arr, right_start + 1, high, comp);
            }
            if (left_size == 0 || left_end == 0) {
                low = high; // sentinel: terminate loop
                return;
            }
            high = left_end - 1;
        }
    };

    while (low < high) {
        std::size_t current_size = high - low + 1;
        if (current_size <= 16) {
            insertion_sort(arr + low, arr + high + 1, comp);
            return;
        }

        median_of_three(arr, low, high, comp);

        if (!std::is_constant_evaluated()) {
            if constexpr (std::contiguous_iterator<I>) {
                using T = std::iter_value_t<I>;
                if constexpr (::algoat::simd::is_simd_supported_v<T>) {
                    if constexpr (std::is_same_v<Compare, std::less<T>> ||
                                  std::is_same_v<Compare, std::less<>>) {
                        if (current_size > SIMD_THRESHOLD) {
                            auto* raw_ptr = std::to_address(arr + low);
                            const T& pivot = *(arr + high);

                            // Proceed with SIMD partitioning only if the array sample is not an
                            // extreme duplicate sequence (arr[low] == arr[high] == pivot means
                            // sample is identical)
                            if (*(arr + low) != pivot) {
                                auto split_opt = ::algoat::simd::partition_simd(
                                    raw_ptr, current_size - 1, pivot);

                                // If SIMD partition succeeded, apply split and advance;
                                // if std::nullopt (scalar host fallback), fall through to
                                // partition_3way.
                                if (split_opt.has_value()) {
                                    std::size_t split_idx = low + *split_opt;
                                    std::ranges::iter_swap(arr + split_idx, arr + high);
                                    recurse_smaller_and_advance(split_idx, split_idx);
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }

        auto [lt, gt] = partition_3way(arr, low, high, comp);
        recurse_smaller_and_advance(lt, gt);
    }
}

} // namespace detail

/**
 * @brief Sorts a range delimited by iterators [first, last) in-place using 3-way partitioning
 * quicksort with custom comparator and projection.
 *
 * @tparam I Random-access iterator type.
 * @tparam S Sentinel type for iterator I.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param first Beginning of the range to sort in-place.
 * @param last Sentinel or end of the range.
 * @param comp Strict weak ordering comparator.
 * @param proj Projection callable.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
constexpr void quicksort(I first, S last, Comp comp = {}, Proj proj = {}) {
    auto last_it = std::ranges::next(first, last);
    auto len = last_it - first;
    if (len <= 1) {
        return;
    }
    auto comp_proj = ::algoat::detail::make_comp_proj(comp, proj);
    detail::quicksort_impl(first, 0, static_cast<std::size_t>(len - 1), comp_proj);
}

/**
 * @brief Sorts a C++20 random-access range in-place using 3-way partitioning quicksort with custom
 * comparator and projection.
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param data Range of elements to sort in-place.
 * @param comp Strict weak ordering comparator.
 * @param proj Projection callable.
 * @return Iterator pointing to the end of the range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> quicksort(R&& data, Comp comp = {}, Proj proj = {}) {
    quicksort(std::ranges::begin(data), std::ranges::end(data), comp, proj);
    return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
}

/**
 * @brief Sorts the span in-place using 3-way partitioning quicksort with a custom comparator.
 * @tparam T Element type.
 * @tparam Compare Comparator callable defining strict weak ordering.
 *
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare> constexpr void quicksort(std::span<T> data, Compare comp) {
    if (data.size() <= 1)
        return;
    detail::quicksort_impl(data.data(), 0, data.size() - 1, comp);
}

/**
 * @brief Sorts the span in-place using 3-way partitioning quicksort.
 *
 * @tparam T Element type supporting @c operator<.
 * @param data Contiguous span of elements to sort.
 */
template <typename T> constexpr void quicksort(std::span<T> data) {
    quicksort(data, std::less<T>{});
}

/**
 * @struct QuickSort
 * @brief Divide-and-conquer sorting algorithm using Dijkstra 3-Way Fat Partitioning (Dutch National
 * Flag).
 *
 * Partitions array ranges into three contiguous slices: @c [ < pivot | == pivot | > pivot ].
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
     *
     * @return "quicksort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "quicksort";
    }

    /**
     * @brief Sorts the span in-place using 3-way partitioning quicksort.
     *
     * @tparam T Element type supporting @c operator<.
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> constexpr void sort(std::span<T> data) const {
        quicksort(data);
    }

    /**
     * @brief Sorts the span in-place using 3-way partitioning quicksort with a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator callable.
     * @param data Contiguous span of elements to sort.
     * @param comp Strict weak ordering comparator.
     */
    template <typename T, typename Compare>
    constexpr void sort(std::span<T> data, Compare comp) const {
        quicksort(data, comp);
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 32 (arrays smaller than 32 elements are better sorted via InsertionSort).
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 32;
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "quicksort", ::algoat::sorting::QuickSort)

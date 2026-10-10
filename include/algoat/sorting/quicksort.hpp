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
 */
template <typename T, typename Compare>
void median_of_three(T* arr, std::size_t low, std::size_t high, Compare comp) {
    std::size_t mid = low + (high - low) / 2;
    if (comp(arr[mid], arr[low]))
        std::swap(arr[low], arr[mid]);
    if (comp(arr[high], arr[low]))
        std::swap(arr[low], arr[high]);
    if (comp(arr[high], arr[mid]))
        std::swap(arr[mid], arr[high]);
    // Place the median at arr[high] so it is excluded from the active partition scan [low, high -
    // 1]
    std::swap(arr[mid], arr[high]);
}

/**
 * @brief Dijkstra 3-Way Fat Partitioning (Dutch National Flag) with zero-copy pivot reference.
 */
template <typename T, typename Compare>
PartitionResult partition_3way(T* arr, std::size_t low, std::size_t high, Compare comp) {
    const T& pivot = arr[high];
    std::size_t lt = low;
    std::size_t i = low;
    std::size_t gt = high - 1;

    while (i <= gt) {
        if (comp(arr[i], pivot)) {
            std::swap(arr[lt], arr[i]);
            ++lt;
            ++i;
        } else if (comp(pivot, arr[i])) {
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
 */
template <typename T, typename Compare>
void quicksort_impl(T* arr, std::size_t low, std::size_t high, Compare comp) {
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
            insertionsort(std::span<T>{arr + low, current_size}, comp);
            return;
        }

        median_of_three(arr, low, high, comp);

        if constexpr (::algoat::simd::is_simd_supported_v<T>) {
            if constexpr (std::is_same_v<Compare, std::less<T>> ||
                          std::is_same_v<Compare, std::less<>>) {
                if (current_size > SIMD_THRESHOLD) {
                    const T& pivot = arr[high];

                    // Proceed with SIMD partitioning only if the array sample is not an extreme
                    // duplicate sequence (arr[low] == arr[high] == pivot means sample is identical)
                    if (arr[low] != pivot) {
                        auto split_opt =
                            ::algoat::simd::partition_simd(arr + low, current_size - 1, pivot);

                        // If SIMD partition succeeded, apply split and advance;
                        // if std::nullopt (scalar host fallback), fall through to partition_3way.
                        if (split_opt.has_value()) {
                            std::size_t split_idx = low + *split_opt;
                            std::swap(arr[split_idx], arr[high]);
                            recurse_smaller_and_advance(split_idx, split_idx);
                            continue;
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
    if constexpr (std::ranges::contiguous_range<R> && std::ranges::sized_range<R>) {
        auto span_data = ::algoat::detail::to_span(data);
        if (span_data.size() <= 1) {
            return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
        }
        if constexpr (std::is_same_v<Proj, std::identity>) {
            detail::quicksort_impl(span_data.data(), 0, span_data.size() - 1, comp);
        } else {
            auto comp_proj = ::algoat::detail::make_comp_proj(comp, proj);
            detail::quicksort_impl(span_data.data(), 0, span_data.size() - 1, comp_proj);
        }
        return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
    } else {
        return ::algoat::detail::fallback_sort(data, std::move(comp), std::move(proj));
    }
}

/**
 * @brief Sorts the span in-place using 3-way partitioning quicksort with a custom comparator.
 * @tparam T Element type.
 * @tparam Compare Comparator callable defining strict weak ordering.
 *
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare> void quicksort(std::span<T> data, Compare comp) {
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
template <typename T> void quicksort(std::span<T> data) {
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
    template <typename T> void sort(std::span<T> data) const {
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
    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
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

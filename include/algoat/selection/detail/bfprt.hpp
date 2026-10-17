/**
 * @file bfprt.hpp
 * @brief Blum-Floyd-Pratt-Rivest-Tarjan (BFPRT 1973) Median-of-Medians algorithm.
 *
 * Provides deterministic worst-case O(N) selection as the fallback engine for Introselect.
 */

#pragma once

#include "algoat/sorting/insertionsort.hpp"

#include <algorithm>
#include <concepts>
#include <functional>
#include <iterator>
#include <ranges>
#include <utility>

namespace algoat::selection::detail {

/**
 * @brief 3-way Dutch National Flag partitioning around a pivot value.
 *
 * Partitions [first, last) into:
 * - [first, low): elements strictly less than pivot
 * - [low, high): elements equivalent to pivot
 * - [high, last): elements strictly greater than pivot
 *
 * @tparam I Random-access iterator type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 * @tparam T Pivot element value type.
 *
 * @param first Beginning of the range to partition.
 * @param last Sentinel or end of the range.
 * @param pivot Pivot value to compare against.
 * @param comp Strict weak ordering comparator callable.
 * @param proj Element projection callable.
 * @return A pair of iterators {low, high} delimiting the range equivalent to pivot.
 */
template <std::random_access_iterator I, typename Comp, typename Proj, typename T>
    requires std::sortable<I, Comp, Proj>
constexpr std::pair<I, I> partition_3way(I first, I last, const T& pivot, Comp comp, Proj proj) {
    I low = first;
    I mid = first;
    I high = last;

    while (mid < high) {
        if (std::invoke(comp, std::invoke(proj, *mid), pivot)) {
            std::ranges::iter_swap(low, mid);
            ++low;
            ++mid;
        } else if (std::invoke(comp, pivot, std::invoke(proj, *mid))) {
            --high;
            std::ranges::iter_swap(mid, high);
        } else {
            ++mid;
        }
    }
    return {low, high};
}

/**
 * @brief Deterministic linear-time BFPRT (Median-of-Medians) selection.
 *
 * Divides the range into groups of 5, recursively selects the median of medians,
 * and 3-way partitions around it. Guaranteed worst-case O(N) comparisons.
 *
 * @tparam I Random-access iterator type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param first Beginning of the subrange.
 * @param nth Iterator pointing to the order statistic to select.
 * @param last Sentinel or end of the subrange.
 * @param comp Strict weak ordering comparator callable.
 * @param proj Element projection callable.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
    requires std::sortable<I, Comp, Proj>
constexpr void bfprt_select(I first, I nth, I last, Comp comp, Proj proj) {
    while (last - first > 16) {
        auto len = last - first;
        auto num_groups = len / 5;

        // Sort each group of 5 and pack medians into prefix [first, first + num_groups)
        for (decltype(num_groups) g = 0; g < num_groups; ++g) {
            I group_first = first + g * 5;
            sorting::insertion_sort(group_first, group_first + 5, comp, proj);
            std::ranges::iter_swap(first + g, group_first + 2);
        }

        // Recursively find median-of-medians
        I medians_first = first;
        I medians_last = first + num_groups;
        I medians_mid = medians_first + num_groups / 2;
        bfprt_select(medians_first, medians_mid, medians_last, comp, proj);

        // Partition around median-of-medians
        auto pivot = std::invoke(proj, *medians_mid);
        auto [mid_first, mid_last] = partition_3way(first, last, pivot, comp, proj);

        if (nth < mid_first) {
            last = mid_first;
        } else if (nth >= mid_last) {
            first = mid_last;
        } else {
            return;
        }
    }
    sorting::insertion_sort(first, last, comp, proj);
}

} // namespace algoat::selection::detail

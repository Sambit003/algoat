/**
 * @file pdqselect.hpp
 * @brief Block Pattern-Defeating Quickselect (PDQSelect) implementation with Branchless Block
 * Partitioning and CMOV register swapping.
 *
 * Implements BlockQuickselect based on Stefan Edelkamp & Armin Weiß (ACM JEA 2019) combined with
 * Orson Peters' (2015) pattern-defeating heuristics and Alexandrescu (2017) adaptive selection.
 * Features branchless block partitioning (B = 128), L1-resident offset buffering, 3-way equal key
 * handling in O(N), Tukey ninther pivot selection, and guaranteed O(N) worst-case time via BFPRT
 * median-of-medians fallback.
 */

#pragma once

#include "algoat/selection/detail/bfprt.hpp"
#include "algoat/sorting/detail/pdq_common.hpp"

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

namespace algoat::selection {

namespace detail {

inline constexpr std::size_t pdqselect_insertion_sort_threshold = 24;
inline constexpr std::size_t pdqselect_ninther_threshold = 128;
inline constexpr std::size_t pdqselect_block_size = 128;

/**
 * @brief Core iterative driver loop for pattern-defeating quickselect.
 *
 * Implements unilateral recursion into only the subarray containing the target order statistic.
 */
template <bool Branchless, std::random_access_iterator I, typename Comp, typename Proj>
constexpr void pdqselect_loop(I begin, I nth, I end, Comp comp, Proj proj, int bad_allowed,
                              bool leftmost = true) {
    using namespace ::algoat::detail::pdq;

    while (true) {
        auto size = end - begin;

        if (static_cast<std::size_t>(size) < pdqselect_insertion_sort_threshold) {
            if (leftmost) {
                pdq_insertion_sort(begin, end, comp, proj);
            } else {
                pdq_unguarded_insertion_sort(begin, end, comp, proj);
            }
            return;
        }

        // Fastpaths for extremum order statistics (k = 0 and k = N - 1)
        if (nth == begin) {
            auto min_it = begin;
            for (auto it = begin + 1; it != end; ++it) {
                if (std::invoke(comp, std::invoke(proj, *it), std::invoke(proj, *min_it))) {
                    min_it = it;
                }
            }
            std::ranges::iter_swap(begin, min_it);
            return;
        }
        if (nth == end - 1) {
            auto max_it = begin;
            for (auto it = begin + 1; it != end; ++it) {
                if (std::invoke(comp, std::invoke(proj, *max_it), std::invoke(proj, *it))) {
                    max_it = it;
                }
            }
            std::ranges::iter_swap(end - 1, max_it);
            return;
        }

        // Pivot selection
        auto s2 = size / 2;
        if (static_cast<std::size_t>(size) > pdqselect_ninther_threshold) {
            pdq_sort3(begin, begin + s2, end - 1, comp, proj);
            pdq_sort3(begin + 1, begin + (s2 - 1), end - 2, comp, proj);
            pdq_sort3(begin + 2, begin + (s2 + 1), end - 3, comp, proj);
            pdq_sort3(begin + (s2 - 1), begin + s2, begin + (s2 + 1), comp, proj);
            std::ranges::iter_swap(begin, begin + s2);
        } else {
            pdq_sort3(begin + s2, begin, end - 1, comp, proj);
        }

        // Duplicate handling: if pivot matches predecessor, all <= pivot are identical
        if (!leftmost &&
            !std::invoke(comp, std::invoke(proj, *(begin - 1)), std::invoke(proj, *begin))) {
            I pivot_pos = pdq_partition_left(begin, end, comp, proj);
            if (nth <= pivot_pos) {
                // nth is within the duplicate pivot run and already correctly positioned
                return;
            }
            begin = pivot_pos + 1;
            continue;
        }

        std::pair<I, bool> part_result;
        if constexpr (Branchless) {
            part_result = pdq_partition_right_branchless<pdqselect_block_size>(begin, end, comp);
        } else {
            part_result = pdq_partition_right(begin, end, comp, proj);
        }
        I pivot_pos = part_result.first;
        bool already_partitioned = part_result.second;

        if (pivot_pos == nth) {
            return;
        }

        auto l_size = pivot_pos - begin;
        auto r_size = end - (pivot_pos + 1);
        bool highly_unbalanced = (l_size < size / 8) || (r_size < size / 8);

        if (highly_unbalanced) {
            if (--bad_allowed == 0) {
                // Worst-case linear time BFPRT fallback guarantees O(N)
                bfprt_select(begin, nth, end, comp, proj);
                return;
            }

            // Shuffle only the active partition containing nth to break pathological inputs
            if (nth < pivot_pos) {
                if (static_cast<std::size_t>(l_size) >= pdqselect_insertion_sort_threshold) {
                    std::ranges::iter_swap(begin, begin + l_size / 4);
                    std::ranges::iter_swap(pivot_pos - 1, pivot_pos - l_size / 4);

                    if (static_cast<std::size_t>(l_size) > pdqselect_ninther_threshold) {
                        std::ranges::iter_swap(begin + 1, begin + (l_size / 4 + 1));
                        std::ranges::iter_swap(begin + 2, begin + (l_size / 4 + 2));
                        std::ranges::iter_swap(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
                        std::ranges::iter_swap(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
                    }
                }
            } else {
                if (static_cast<std::size_t>(r_size) >= pdqselect_insertion_sort_threshold) {
                    std::ranges::iter_swap(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
                    std::ranges::iter_swap(end - 1, end - r_size / 4);

                    if (static_cast<std::size_t>(r_size) > pdqselect_ninther_threshold) {
                        std::ranges::iter_swap(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
                        std::ranges::iter_swap(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
                        std::ranges::iter_swap(end - 2, end - (1 + r_size / 4));
                        std::ranges::iter_swap(end - 3, end - (2 + r_size / 4));
                    }
                }
            }
        } else {
            if (already_partitioned) {
                if (nth < pivot_pos) {
                    if (pdq_partial_insertion_sort(begin, pivot_pos, comp, proj)) {
                        return;
                    }
                } else {
                    if (pdq_partial_insertion_sort(pivot_pos + 1, end, comp, proj)) {
                        return;
                    }
                }
            }
        }

        // Unilateral recursion
        if (nth < pivot_pos) {
            end = pivot_pos;
        } else {
            begin = pivot_pos + 1;
            leftmost = false;
        }
    }
}

} // namespace detail

/**
 * @brief Selects the nth order statistic in-place using Block Pattern-Defeating Quickselect.
 *
 * Rearranges elements such that the element at @c nth is the one that would be in that position
 * if [first, last) were sorted according to @c comp and @c proj. Elements before @c nth are
 * not greater than @c *nth, and elements after @c nth are not less than @c *nth.
 *
 * Employs branchless BlockQuicksort partitioning (Edelkamp & Weiß 2019), pattern-defeating
 * heuristics (Peters 2015), and worst-case O(N) introspective BFPRT fallback.
 *
 * @tparam I Random-access iterator type satisfying std::sortable.
 * @tparam S Sentinel type for iterator I.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param first Beginning of the range.
 * @param nth Iterator pointing to the target order statistic.
 * @param last Sentinel or end of the range.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Sentinel iterator pointing to past-the-end of the range, matching
 * std::ranges::nth_element convention.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
constexpr I pdqselect(I first, I nth, S last, Comp comp = {}, Proj proj = {}) {
    auto last_it = std::ranges::next(first, last);
    if (first == last_it || nth == last_it) {
        return last_it;
    }
    auto len = last_it - first;
    if (len <= 1) {
        return last_it;
    }

    int bad_allowed = 2 * std::bit_width(static_cast<std::size_t>(len));

    if constexpr (::algoat::detail::pdq::is_branchless_comparable_v<I, Comp, Proj>) {
        detail::pdqselect_loop<true>(first, nth, last_it, comp, proj, bad_allowed);
    } else {
        detail::pdqselect_loop<false>(first, nth, last_it, comp, proj, bad_allowed);
    }
    return last_it;
}

/**
 * @brief Range overload for @c algoat::selection::pdqselect.
 *
 * @tparam R Random-access range type satisfying std::sortable.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param data Range of elements to select within in-place.
 * @param nth Iterator pointing to the target order statistic.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Sentinel iterator pointing to past-the-end of the range, matching
 * std::ranges::nth_element convention.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> pdqselect(R&& data, std::ranges::iterator_t<R> nth,
                                                        Comp comp = {}, Proj proj = {}) {
    return pdqselect(std::ranges::begin(data), nth, std::ranges::end(data), std::move(comp),
                     std::move(proj));
}

/**
 * @brief Contiguous std::span overload for @c algoat::selection::pdqselect.
 *
 * @tparam T Element type.
 * @tparam Extent Extent of the span (supports dynamic and static extents).
 * @tparam Comp Strict weak ordering comparator callable.
 *
 * @param data Contiguous span of elements.
 * @param k Rank index to select in-place.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, std::size_t Extent = std::dynamic_extent, typename Comp = std::ranges::less>
constexpr void pdqselect(std::span<T, Extent> data, std::size_t k, Comp comp = {}) {
    if (k >= data.size()) {
        return;
    }
    pdqselect(data.begin(), data.begin() + k, data.end(), std::move(comp), std::identity{});
}

} // namespace algoat::selection

/**
 * @file multi_select.hpp
 * @brief Optimal Multi-Selection and Quantile Selection Engine in O(N log M).
 *
 * Implements optimal multiple selection matching the Kaligosi-Mehlhorn-Munro-Sanders (ICALP 2005)
 * information-theoretic lower bound and Chambers Algorithm 410 via recursive median-rank bisection
 * powered by Floyd-Rivest Algorithm 489.
 */

#pragma once

#include "algoat/selection/nth_element.hpp"
#include "algoat/selection/select_optimal.hpp"
#include "algoat/sorting/insertionsort.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::selection {

namespace detail {

/**
 * @brief Recursive median-rank bisection divide-and-conquer engine.
 *
 * Partitions array slice [first + left, first + right + 1) around the median query rank,
 * and recursively partitions left and right subarrays for remaining ranks.
 *
 * @tparam I Random-access iterator type.
 * @tparam RanksRange Random-access range of sorted rank indices.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Element projection callable.
 */
template <std::random_access_iterator I, typename RanksRange, typename Comp, typename Proj>
constexpr void multi_select_recurse(I first, std::ptrdiff_t left, std::ptrdiff_t right,
                                    const RanksRange& ranks, std::ptrdiff_t rank_left,
                                    std::ptrdiff_t rank_right, Comp comp, Proj proj) {
    if (rank_left > rank_right || left >= right) {
        return;
    }

    if (right - left <= 32) {
        sorting::insertion_sort(first + left, first + right + 1, comp, proj);
        return;
    }

    // Base case: exactly one rank in this partition
    if (rank_left == rank_right) {
        const auto target_r = static_cast<std::ptrdiff_t>(ranks[rank_left]);
        if (target_r >= left && target_r <= right) {
            select_optimal(first + left, first + target_r, first + right + 1, comp, proj);
        }
        return;
    }

    // Optimization: exactly two ranks in this partition
    if (rank_right - rank_left == 1) {
        const auto r1 = static_cast<std::ptrdiff_t>(ranks[rank_left]);
        const auto r2 = static_cast<std::ptrdiff_t>(ranks[rank_right]);
        if (r1 >= left && r1 <= right) {
            select_optimal(first + left, first + r1, first + right + 1, comp, proj);
        }
        if (r2 > r1 && r2 <= right) {
            select_optimal(first + r1 + 1, first + r2, first + right + 1, comp, proj);
        }
        return;
    }

    // Bisect query ranks at median rank index
    const std::ptrdiff_t m = rank_left + (rank_right - rank_left) / 2;
    const auto target_r = static_cast<std::ptrdiff_t>(ranks[m]);

    if (target_r >= left && target_r <= right) {
        select_optimal(first + left, first + target_r, first + right + 1, comp, proj);
    }

    // Suppress duplicates matching the pivot rank
    std::ptrdiff_t m_left = m - 1;
    while (m_left >= rank_left && static_cast<std::ptrdiff_t>(ranks[m_left]) == target_r) {
        --m_left;
    }

    std::ptrdiff_t m_right = m + 1;
    while (m_right <= rank_right && static_cast<std::ptrdiff_t>(ranks[m_right]) == target_r) {
        ++m_right;
    }

    // Recurse on left subarray if there are remaining target ranks
    if (m_left >= rank_left && target_r > left) {
        multi_select_recurse(first, left, target_r - 1, ranks, rank_left, m_left, comp, proj);
    }

    // Recurse on right subarray if there are remaining target ranks
    if (m_right <= rank_right && target_r < right) {
        multi_select_recurse(first, target_r + 1, right, ranks, m_right, rank_right, comp, proj);
    }
}

} // namespace detail

/**
 * @brief Selects multiple arbitrary sorted order statistics in O(N log M) time.
 *
 * Rearranges elements in [first, last) such that for each rank k in @c ranks,
 * *(first + k) is placed into its exact global sorted position, all preceding elements
 * satisfy !(comp(*(first + k), *it)), and all succeeding elements satisfy !(comp(*it, *(first +
 * k))).
 *
 * @tparam I Random-access iterator type satisfying sortable.
 * @tparam S Sentinel type for I.
 * @tparam RanksRange Random-access range of sorted integral ranks.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Element projection callable.
 *
 * @param first Beginning of the range to partition.
 * @param last Sentinel or past-the-end iterator.
 * @param ranks Random-access range of sorted rank indices (0 <= k < N).
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Past-the-end iterator @c last.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S,
          std::ranges::random_access_range RanksRange, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
constexpr I multi_select(I first, S last, RanksRange&& ranks, Comp comp = {}, Proj proj = {}) {
    auto last_it = std::ranges::next(first, last);
    const auto n = static_cast<std::ptrdiff_t>(std::distance(first, last_it));
    const auto m = static_cast<std::ptrdiff_t>(std::ranges::distance(ranks));

    if (n <= 1 || m == 0) {
        return last_it;
    }

    if (std::is_constant_evaluated()) {
        for (auto r : ranks) {
            const auto target_r = static_cast<std::ptrdiff_t>(r);
            if (target_r >= 0 && target_r < n) {
                nth_element(first, first + target_r, last_it, comp, proj);
            }
        }
        return last_it;
    }

    detail::multi_select_recurse(first, 0, n - 1, ranks, 0, m - 1, comp, proj);
    return last_it;
}

/**
 * @brief Range overload for @c multi_select.
 *
 * Rearranges elements in @c data such that all order statistics at indices in @c ranks
 * are positioned in their exact global sorted order.
 *
 * @tparam R Random-access range type.
 * @tparam RanksRange Random-access range of sorted rank indices.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Element projection callable.
 *
 * @param data Random-access range to rearrange.
 * @param ranks Range of sorted rank indices.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Borrowed iterator to the end of @c data.
 */
template <std::ranges::random_access_range R, std::ranges::random_access_range RanksRange,
          typename Comp = std::ranges::less, typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> multi_select(R&& data, RanksRange&& ranks,
                                                           Comp comp = {}, Proj proj = {}) {
    return multi_select(std::ranges::begin(data), std::ranges::end(data), ranks, std::move(comp),
                        std::move(proj));
}

/**
 * @brief High-level quantile helper: extracts Q evenly spaced quantiles.
 *
 * Extracts the Q - 1 cutpoints separating Q equal-sized intervals (e.g. Q = 4 extracts
 * quartiles Q1, median, Q3; Q = 10 extracts deciles p10, p20, ..., p90).
 * Employs a zero-heap-allocation lazy transform view to evaluate rank offsets on-the-fly,
 * strictly bounding memory consumption to O(log Q) recursion stack.
 *
 * @tparam T Element value type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Element projection callable.
 *
 * @param data Contiguous span of elements (rearranged partially in-place).
 * @param num_quantiles Number of intervals (Q >= 2).
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Vector of extracted quantile values in non-decreasing order.
 */
template <typename T, typename Comp = std::ranges::less, typename Proj = std::identity>
[[nodiscard]] constexpr std::vector<T> quantiles(std::span<T> data, std::size_t num_quantiles,
                                                 Comp comp = {}, Proj proj = {}) {
    if (num_quantiles <= 1 || data.empty()) {
        return {};
    }

    const auto n = static_cast<std::ptrdiff_t>(data.size());
    const auto m = static_cast<std::ptrdiff_t>(num_quantiles - 1);
    const auto total_q = static_cast<std::ptrdiff_t>(num_quantiles);

    // Zero-allocation lazy transform view over rank indices [0, m)
    auto ranks_view = std::views::iota(std::ptrdiff_t{0}, m) |
                      std::views::transform([n, total_q](std::ptrdiff_t i) -> std::ptrdiff_t {
                          const std::ptrdiff_t q = i + 1;
                          return (q * (n - 1)) / total_q;
                      });

    multi_select(data.begin(), data.end(), ranks_view, comp, proj);

    std::vector<T> result;
    result.reserve(static_cast<std::size_t>(m));
    for (std::ptrdiff_t r : ranks_view) {
        result.push_back(data[static_cast<std::size_t>(r)]);
    }
    return result;
}

} // namespace algoat::selection

/**
 * @file pdqsort.hpp
 * @brief Pattern-Defeating Quicksort (PDQSort) implementation with BlockQuicksort branchless
 * partitioning.
 *
 * Implements Orson Peters' (2015) PDQSort algorithm combining fast average-case randomized
 * quicksort, worst-case O(N log N) heapsort fallback, branchless BlockQuicksort partitioning
 * (Edelkamp & Weiß 2016), and linear-time O(N) adaptation for sorted, reverse-sorted, and
 * equal-element sequences.
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/sorting/detail/pdq_common.hpp"
#include "algoat/sorting/heapsort.hpp"

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace algoat::sorting {

namespace detail {

inline constexpr std::size_t pdq_insertion_sort_threshold = 24;
inline constexpr std::size_t pdq_ninther_threshold = 128;
inline constexpr std::size_t pdq_partial_insertion_sort_limit = 8;
inline constexpr std::size_t pdq_block_size = 64;

using ::algoat::detail::pdq::is_branchless_comparable_v;
template <typename I, typename Comp, typename Proj>
inline constexpr bool is_branchless_sortable_v = is_branchless_comparable_v<I, Comp, Proj>;

using ::algoat::detail::pdq::pdq_insertion_sort;
using ::algoat::detail::pdq::pdq_partial_insertion_sort;
using ::algoat::detail::pdq::pdq_partition_left;
using ::algoat::detail::pdq::pdq_partition_right;
using ::algoat::detail::pdq::pdq_sort2;
using ::algoat::detail::pdq::pdq_sort3;
using ::algoat::detail::pdq::pdq_swap_offsets;
using ::algoat::detail::pdq::pdq_unguarded_insertion_sort;
using ::algoat::detail::pdq::scan_partition_block;
using ::algoat::detail::pdq::ScanDirection;

template <std::random_access_iterator I, typename Comp>
constexpr std::pair<I, bool> pdq_partition_right_branchless(I begin, I end, Comp& comp) {
    return ::algoat::detail::pdq::pdq_partition_right_branchless<pdq_block_size>(begin, end, comp);
}

/**
 * @brief Core recursive driver loop for pattern-defeating quicksort.
 */
template <bool Branchless, std::random_access_iterator I, typename Comp, typename Proj>
constexpr void pdqsort_loop(I begin, I end, Comp comp, Proj proj, int bad_allowed,
                            bool leftmost = true) {
    while (true) {
        auto size = end - begin;

        if (static_cast<std::size_t>(size) < pdq_insertion_sort_threshold) {
            if (leftmost) {
                pdq_insertion_sort(begin, end, comp, proj);
            } else {
                pdq_unguarded_insertion_sort(begin, end, comp, proj);
            }
            return;
        }

        auto s2 = size / 2;
        if (static_cast<std::size_t>(size) > pdq_ninther_threshold) {
            pdq_sort3(begin, begin + s2, end - 1, comp, proj);
            pdq_sort3(begin + 1, begin + (s2 - 1), end - 2, comp, proj);
            pdq_sort3(begin + 2, begin + (s2 + 1), end - 3, comp, proj);
            pdq_sort3(begin + (s2 - 1), begin + s2, begin + (s2 + 1), comp, proj);
            std::ranges::iter_swap(begin, begin + s2);
        } else {
            pdq_sort3(begin + s2, begin, end - 1, comp, proj);
        }

        if (!leftmost &&
            !std::invoke(comp, std::invoke(proj, *(begin - 1)), std::invoke(proj, *begin))) {
            begin = pdq_partition_left(begin, end, comp, proj) + 1;
            continue;
        }

        std::pair<I, bool> part_result;
        if constexpr (Branchless) {
            part_result = pdq_partition_right_branchless(begin, end, comp);
        } else {
            part_result = pdq_partition_right(begin, end, comp, proj);
        }
        I pivot_pos = part_result.first;
        bool already_partitioned = part_result.second;

        auto l_size = pivot_pos - begin;
        auto r_size = end - (pivot_pos + 1);
        bool highly_unbalanced = (l_size < size / 8) || (r_size < size / 8);

        if (highly_unbalanced) {
            if (--bad_allowed == 0) {
                heapsort(begin, end, comp, proj);
                return;
            }

            if (static_cast<std::size_t>(l_size) >= pdq_insertion_sort_threshold) {
                std::ranges::iter_swap(begin, begin + l_size / 4);
                std::ranges::iter_swap(pivot_pos - 1, pivot_pos - l_size / 4);

                if (static_cast<std::size_t>(l_size) > pdq_ninther_threshold) {
                    std::ranges::iter_swap(begin + 1, begin + (l_size / 4 + 1));
                    std::ranges::iter_swap(begin + 2, begin + (l_size / 4 + 2));
                    std::ranges::iter_swap(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
                    std::ranges::iter_swap(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
                }
            }

            if (static_cast<std::size_t>(r_size) >= pdq_insertion_sort_threshold) {
                std::ranges::iter_swap(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
                std::ranges::iter_swap(end - 1, end - r_size / 4);

                if (static_cast<std::size_t>(r_size) > pdq_ninther_threshold) {
                    std::ranges::iter_swap(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
                    std::ranges::iter_swap(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
                    std::ranges::iter_swap(end - 2, end - (1 + r_size / 4));
                    std::ranges::iter_swap(end - 3, end - (2 + r_size / 4));
                }
            }
        } else {
            if (already_partitioned && pdq_partial_insertion_sort(begin, pivot_pos, comp, proj) &&
                pdq_partial_insertion_sort(pivot_pos + 1, end, comp, proj)) {
                return;
            }
        }

        pdqsort_loop<Branchless>(begin, pivot_pos, comp, proj, bad_allowed, leftmost);
        begin = pivot_pos + 1;
        leftmost = false;
    }
}

} // namespace detail

/**
 * @brief Sorts the iterator range [first, last) in-place using Pattern-Defeating Quicksort.
 *
 * @tparam I Random-access iterator type satisfying std::sortable.
 * @tparam S Sentinel type for iterator I.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param first Beginning of the range to sort.
 * @param last Sentinel or end of the range.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
constexpr void pdqsort(I first, S last, Comp comp = {}, Proj proj = {}) {
    auto last_it = std::ranges::next(first, last);
    auto len = last_it - first;
    if (len <= 1) {
        return;
    }

    int bad_allowed = std::bit_width(static_cast<std::size_t>(len));

    if constexpr (detail::is_branchless_sortable_v<I, Comp, Proj>) {
        detail::pdqsort_loop<true>(first, last_it, comp, proj, bad_allowed);
    } else {
        detail::pdqsort_loop<false>(first, last_it, comp, proj, bad_allowed);
    }
}

/**
 * @brief Sorts a C++20 random-access range in-place using Pattern-Defeating Quicksort.
 *
 * @tparam R Random-access range type satisfying std::sortable.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param r Range of elements to sort in-place.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Iterator to the end of the range, or std::ranges::dangling if passed an rvalue
 * non-borrowed range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> pdqsort(R&& r, Comp comp = {}, Proj proj = {}) {
    pdqsort(std::ranges::begin(r), std::ranges::end(r), std::move(comp), std::move(proj));
    return std::ranges::next(std::ranges::begin(r), std::ranges::end(r));
}

/**
 * @brief Sorts a contiguous span in-place using Pattern-Defeating Quicksort.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare = std::ranges::less>
constexpr void pdqsort(std::span<T> data, Compare comp = {}) {
    pdqsort(data.begin(), data.end(), std::move(comp), std::identity{});
}

/**
 * @struct PdqSort
 * @brief Zero-overhead adapter struct for Pattern-Defeating Quicksort (PDQSort).
 *
 * Satisfies the @c algoat::sorting::SortAlgorithm concept for static polymorphism
 * and registry integration.
 */
struct PdqSort {
    /**
     * @brief Unique algorithm name identifier.
     * @return "pdqsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "pdqsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts the span in-place using default comparator.
     * @tparam T Element type.
     * @param data Contiguous span of elements.
     */
    template <typename T> constexpr void sort(std::span<T> data) const {
        pdqsort(data);
    }

    /**
     * @brief Sorts the span in-place using a custom comparator.
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator.
     * @param data Contiguous span of elements.
     * @param comp Comparator callable.
     */
    template <typename T, typename Compare>
    constexpr void sort(std::span<T> data, Compare comp) const {
        pdqsort(data, std::move(comp));
    }
};

using PDQSort = PdqSort;

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "pdqsort", ::algoat::sorting::PdqSort)

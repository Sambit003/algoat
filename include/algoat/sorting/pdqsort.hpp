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

/**
 * @brief Projected comparison adapter callable for C++20 projections and comparators.
 */
template <typename Comp, typename Proj> struct ProjectedCompare {
    [[no_unique_address]] Comp comp;
    [[no_unique_address]] Proj proj;

    template <typename T, typename U> constexpr bool operator()(T&& a, U&& b) const {
        return std::invoke(comp, std::invoke(proj, std::forward<T>(a)),
                           std::invoke(proj, std::forward<U>(b)));
    }
};

/**
 * @brief Trait to determine whether branchless block partitioning should be enabled.
 */
template <typename I, typename Comp, typename Proj>
inline constexpr bool is_branchless_sortable_v =
    std::is_arithmetic_v<std::iter_value_t<I>> &&
    (std::is_same_v<Comp, std::ranges::less> || std::is_same_v<Comp, std::less<>> ||
     std::is_same_v<Comp, std::less<std::iter_value_t<I>>> ||
     std::is_same_v<Comp, std::ranges::greater> || std::is_same_v<Comp, std::greater<>> ||
     std::is_same_v<Comp, std::greater<std::iter_value_t<I>>>) &&
    std::is_same_v<Proj, std::identity>;

/**
 * @brief Sorts two iterators if they are out of order according to the comparator.
 */
template <std::random_access_iterator I, typename Compare>
constexpr void pdq_sort2(I a, I b, Compare comp) {
    if (comp(*b, *a)) {
        std::ranges::iter_swap(a, b);
    }
}

/**
 * @brief Sorts three iterators in-place using at most 3 comparisons.
 */
template <std::random_access_iterator I, typename Compare>
constexpr void pdq_sort3(I a, I b, I c, Compare comp) {
    pdq_sort2(a, b, comp);
    pdq_sort2(b, c, comp);
    pdq_sort2(a, b, comp);
}

/**
 * @brief Standard guarded insertion sort.
 */
template <std::random_access_iterator I, typename Compare>
constexpr void pdq_insertion_sort(I begin, I end, Compare comp) {
    if (begin == end) {
        return;
    }
    for (I cur = begin + 1; cur != end; ++cur) {
        I sift = cur;
        I sift_1 = cur - 1;
        if (comp(*sift, *sift_1)) {
            auto val = std::move(*sift);
            do {
                *sift-- = std::move(*sift_1);
            } while (sift != begin && comp(val, *--sift_1));
            *sift = std::move(val);
        }
    }
}

/**
 * @brief Unguarded insertion sort assuming a sentinel element strictly less than or equal
 * to any element in [begin, end) exists before begin.
 */
template <std::random_access_iterator I, typename Compare>
constexpr void pdq_unguarded_insertion_sort(I begin, I end, Compare comp) {
    if (begin == end) {
        return;
    }
    for (I cur = begin + 1; cur != end; ++cur) {
        I sift = cur;
        I sift_1 = cur - 1;
        if (comp(*sift, *sift_1)) {
            auto val = std::move(*sift);
            do {
                *sift-- = std::move(*sift_1);
            } while (comp(val, *--sift_1));
            *sift = std::move(val);
        }
    }
}

/**
 * @brief Attempts insertion sort, aborting if more than pdq_partial_insertion_sort_limit
 * elements are moved. Used to check for already-sorted or nearly-sorted ranges in O(N).
 */
template <std::random_access_iterator I, typename Compare>
constexpr bool pdq_partial_insertion_sort(I begin, I end, Compare comp) {
    if (begin == end) {
        return true;
    }
    std::size_t limit = 0;
    for (I cur = begin + 1; cur != end; ++cur) {
        I sift = cur;
        I sift_1 = cur - 1;
        if (comp(*sift, *sift_1)) {
            auto val = std::move(*sift);
            do {
                *sift-- = std::move(*sift_1);
            } while (sift != begin && comp(val, *--sift_1));
            *sift = std::move(val);
            limit += static_cast<std::size_t>(cur - sift);
        }
        if (limit > pdq_partial_insertion_sort_limit) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Swaps elements indexed by offset buffers across left and right partitions.
 *
 * @tparam I Random-access iterator type.
 * @param first Base iterator for left block offsets.
 * @param last Base iterator for right block offsets.
 * @param offsets_l Array of relative indices in the left block to swap.
 * @param offsets_r Array of relative indices in the right block to swap.
 * @param num Number of elements to swap in this batch.
 * @param use_swaps If true (e.g. num_l == num_r), performs pairwise element swaps preserving
 * symmetry; otherwise executes cyclical moves.
 */
template <std::random_access_iterator I>
constexpr void pdq_swap_offsets(I first, I last, const std::uint8_t* offsets_l,
                                const std::uint8_t* offsets_r, std::size_t num, bool use_swaps) {
    using T = std::iter_value_t<I>;
    if (use_swaps) {
        for (std::size_t i = 0; i < num; ++i) {
            std::ranges::iter_swap(first + offsets_l[i], last - offsets_r[i]);
        }
    } else if (num > 0) {
        I l = first + offsets_l[0];
        I r = last - offsets_r[0];
        T tmp = std::move(*l);
        *l = std::move(*r);
        for (std::size_t i = 1; i < num; ++i) {
            l = first + offsets_l[i];
            *r = std::move(*l);
            r = last - offsets_r[i];
            *l = std::move(*r);
        }
        *r = std::move(tmp);
    }
}

/**
 * @brief BlockQuicksort branchless partitioning algorithm.
 */
template <std::random_access_iterator I, typename Compare>
constexpr std::pair<I, bool> pdq_partition_right_branchless(I begin, I end, Compare comp) {
    using T = std::iter_value_t<I>;
    T pivot = std::move(*begin);
    I first = begin;
    I last = end;

    while (comp(*++first, pivot)) {
    }

    if (first - 1 == begin) {
        while (first < last && !comp(*--last, pivot)) {
        }
    } else {
        while (!comp(*--last, pivot)) {
        }
    }

    bool already_partitioned = first >= last;
    if (!already_partitioned) {
        std::ranges::iter_swap(first, last);
        ++first;

        alignas(64) std::uint8_t offsets_l[pdq_block_size];
        alignas(64) std::uint8_t offsets_r[pdq_block_size];

        I offsets_l_base = first;
        I offsets_r_base = last;
        std::size_t num_l = 0;
        std::size_t num_r = 0;
        std::size_t start_l = 0;
        std::size_t start_r = 0;

        while (first < last) {
            std::size_t num_unknown = static_cast<std::size_t>(last - first);
            std::size_t left_split = num_l == 0 ? (num_r == 0 ? num_unknown / 2 : num_unknown) : 0;
            std::size_t right_split = num_r == 0 ? (num_unknown - left_split) : 0;

            if (left_split >= pdq_block_size) {
                for (std::size_t i = 0; i < pdq_block_size;) {
                    offsets_l[num_l] = static_cast<std::uint8_t>(i++);
                    num_l += !comp(*first, pivot);
                    ++first;
                    offsets_l[num_l] = static_cast<std::uint8_t>(i++);
                    num_l += !comp(*first, pivot);
                    ++first;
                    offsets_l[num_l] = static_cast<std::uint8_t>(i++);
                    num_l += !comp(*first, pivot);
                    ++first;
                    offsets_l[num_l] = static_cast<std::uint8_t>(i++);
                    num_l += !comp(*first, pivot);
                    ++first;
                    offsets_l[num_l] = static_cast<std::uint8_t>(i++);
                    num_l += !comp(*first, pivot);
                    ++first;
                    offsets_l[num_l] = static_cast<std::uint8_t>(i++);
                    num_l += !comp(*first, pivot);
                    ++first;
                    offsets_l[num_l] = static_cast<std::uint8_t>(i++);
                    num_l += !comp(*first, pivot);
                    ++first;
                    offsets_l[num_l] = static_cast<std::uint8_t>(i++);
                    num_l += !comp(*first, pivot);
                    ++first;
                }
            } else {
                for (std::size_t i = 0; i < left_split;) {
                    offsets_l[num_l] = static_cast<std::uint8_t>(i++);
                    num_l += !comp(*first, pivot);
                    ++first;
                }
            }

            if (right_split >= pdq_block_size) {
                for (std::size_t i = 0; i < pdq_block_size;) {
                    offsets_r[num_r] = static_cast<std::uint8_t>(++i);
                    num_r += comp(*--last, pivot);
                    offsets_r[num_r] = static_cast<std::uint8_t>(++i);
                    num_r += comp(*--last, pivot);
                    offsets_r[num_r] = static_cast<std::uint8_t>(++i);
                    num_r += comp(*--last, pivot);
                    offsets_r[num_r] = static_cast<std::uint8_t>(++i);
                    num_r += comp(*--last, pivot);
                    offsets_r[num_r] = static_cast<std::uint8_t>(++i);
                    num_r += comp(*--last, pivot);
                    offsets_r[num_r] = static_cast<std::uint8_t>(++i);
                    num_r += comp(*--last, pivot);
                    offsets_r[num_r] = static_cast<std::uint8_t>(++i);
                    num_r += comp(*--last, pivot);
                    offsets_r[num_r] = static_cast<std::uint8_t>(++i);
                    num_r += comp(*--last, pivot);
                }
            } else {
                for (std::size_t i = 0; i < right_split;) {
                    offsets_r[num_r] = static_cast<std::uint8_t>(++i);
                    num_r += comp(*--last, pivot);
                }
            }

            std::size_t num = std::min(num_l, num_r);
            pdq_swap_offsets(offsets_l_base, offsets_r_base, offsets_l + start_l,
                             offsets_r + start_r, num, num_l == num_r);
            num_l -= num;
            num_r -= num;
            start_l += num;
            start_r += num;

            if (num_l == 0) {
                start_l = 0;
                offsets_l_base = first;
            }
            if (num_r == 0) {
                start_r = 0;
                offsets_r_base = last;
            }
        }

        if (num_l > 0) {
            std::size_t offset_idx = start_l;
            while (num_l--) {
                std::ranges::iter_swap(offsets_l_base + offsets_l[offset_idx + num_l], --last);
            }
            first = last;
        }
        if (num_r > 0) {
            std::size_t offset_idx = start_r;
            while (num_r--) {
                std::ranges::iter_swap(offsets_r_base - offsets_r[offset_idx + num_r], first);
                ++first;
            }
            last = first;
        }
    }

    I pivot_pos = first - 1;
    *begin = std::move(*pivot_pos);
    *pivot_pos = std::move(pivot);

    return std::make_pair(pivot_pos, already_partitioned);
}

/**
 * @brief Standard branchy partitioning fallback for non-arithmetic or custom-projected types.
 */
template <std::random_access_iterator I, typename Compare>
constexpr std::pair<I, bool> pdq_partition_right(I begin, I end, Compare comp) {
    using T = std::iter_value_t<I>;
    T pivot = std::move(*begin);
    I first = begin;
    I last = end;

    while (comp(*++first, pivot)) {
    }

    if (first - 1 == begin) {
        while (first < last && !comp(*--last, pivot)) {
        }
    } else {
        while (!comp(*--last, pivot)) {
        }
    }

    bool already_partitioned = first >= last;

    while (first < last) {
        std::ranges::iter_swap(first, last);
        while (comp(*++first, pivot)) {
        }
        while (!comp(*--last, pivot)) {
        }
    }

    I pivot_pos = first - 1;
    *begin = std::move(*pivot_pos);
    *pivot_pos = std::move(pivot);

    return std::make_pair(pivot_pos, already_partitioned);
}

/**
 * @brief Left partition putting elements equal to the pivot on the left partition.
 * Collapses duplicate elements into a contiguous block, eliminating them in O(N).
 */
template <std::random_access_iterator I, typename Compare>
constexpr I pdq_partition_left(I begin, I end, Compare comp) {
    using T = std::iter_value_t<I>;
    T pivot = std::move(*begin);
    I first = begin;
    I last = end;

    while (comp(pivot, *--last)) {
    }

    if (last + 1 == end) {
        while (first < last && !comp(pivot, *++first)) {
        }
    } else {
        while (!comp(pivot, *++first)) {
        }
    }

    while (first < last) {
        std::ranges::iter_swap(first, last);
        while (comp(pivot, *--last)) {
        }
        while (!comp(pivot, *++first)) {
        }
    }

    I pivot_pos = last;
    *begin = std::move(*pivot_pos);
    *pivot_pos = std::move(pivot);

    return pivot_pos;
}

/**
 * @brief Core recursive driver loop for pattern-defeating quicksort.
 */
template <bool Branchless, std::random_access_iterator I, typename Compare>
constexpr void pdqsort_loop(I begin, I end, Compare comp, int bad_allowed, bool leftmost = true) {
    while (true) {
        auto size = end - begin;

        if (static_cast<std::size_t>(size) < pdq_insertion_sort_threshold) {
            if (leftmost) {
                pdq_insertion_sort(begin, end, comp);
            } else {
                pdq_unguarded_insertion_sort(begin, end, comp);
            }
            return;
        }

        auto s2 = size / 2;
        if (static_cast<std::size_t>(size) > pdq_ninther_threshold) {
            pdq_sort3(begin, begin + s2, end - 1, comp);
            pdq_sort3(begin + 1, begin + (s2 - 1), end - 2, comp);
            pdq_sort3(begin + 2, begin + (s2 + 1), end - 3, comp);
            pdq_sort3(begin + (s2 - 1), begin + s2, begin + (s2 + 1), comp);
            std::ranges::iter_swap(begin, begin + s2);
        } else {
            pdq_sort3(begin + s2, begin, end - 1, comp);
        }

        if (!leftmost && !comp(*(begin - 1), *begin)) {
            begin = pdq_partition_left(begin, end, comp) + 1;
            continue;
        }

        std::pair<I, bool> part_result = Branchless
                                             ? pdq_partition_right_branchless(begin, end, comp)
                                             : pdq_partition_right(begin, end, comp);
        I pivot_pos = part_result.first;
        bool already_partitioned = part_result.second;

        auto l_size = pivot_pos - begin;
        auto r_size = end - (pivot_pos + 1);
        bool highly_unbalanced = (l_size < size / 8) || (r_size < size / 8);

        if (highly_unbalanced) {
            if (--bad_allowed == 0) {
                heapsort(begin, end, comp, std::identity{});
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
            if (already_partitioned && pdq_partial_insertion_sort(begin, pivot_pos, comp) &&
                pdq_partial_insertion_sort(pivot_pos + 1, end, comp)) {
                return;
            }
        }

        pdqsort_loop<Branchless>(begin, pivot_pos, comp, bad_allowed, leftmost);
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
    detail::ProjectedCompare<Comp, Proj> projected_comp{std::move(comp), std::move(proj)};

    if constexpr (detail::is_branchless_sortable_v<I, Comp, Proj>) {
        detail::pdqsort_loop<true>(first, last_it, projected_comp, bad_allowed);
    } else {
        detail::pdqsort_loop<false>(first, last_it, projected_comp, bad_allowed);
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

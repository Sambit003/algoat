/**
 * @file pdq_common.hpp
 * @brief Common partitioning and small-sort primitives for PDQSort and PDQSelect.
 *
 * Implements BlockQuicksort branchless scanning, 3-element median sorting, unguarded
 * insertion sort, and pattern-defeating duplicate partitioning routines.
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

namespace algoat::detail::pdq {

/**
 * @brief Trait to determine whether branchless block partitioning should be enabled.
 */
template <typename I, typename Comp, typename Proj>
inline constexpr bool is_branchless_comparable_v =
    std::is_arithmetic_v<std::iter_value_t<I>> &&
    (std::is_same_v<Comp, std::ranges::less> || std::is_same_v<Comp, std::less<>> ||
     std::is_same_v<Comp, std::less<std::iter_value_t<I>>> ||
     std::is_same_v<Comp, std::ranges::greater> || std::is_same_v<Comp, std::greater<>> ||
     std::is_same_v<Comp, std::greater<std::iter_value_t<I>>>) &&
    std::is_same_v<Proj, std::identity>;

/**
 * @brief Sorts two iterators if they are out of order according to comparator and projection.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void pdq_sort2(I a, I b, Comp& comp, Proj& proj) {
    if (std::invoke(comp, std::invoke(proj, *b), std::invoke(proj, *a))) {
        std::ranges::iter_swap(a, b);
    }
}

/**
 * @brief Sorts three iterators in-place using at most 3 comparisons.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void pdq_sort3(I a, I b, I c, Comp& comp, Proj& proj) {
    pdq_sort2(a, b, comp, proj);
    pdq_sort2(b, c, comp, proj);
    pdq_sort2(a, b, comp, proj);
}

/**
 * @brief Standard guarded insertion sort.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void pdq_insertion_sort(I begin, I end, Comp& comp, Proj& proj) {
    if (begin == end) {
        return;
    }
    for (I cur = begin + 1; cur != end; ++cur) {
        I sift = cur;
        I sift_1 = cur - 1;
        if (std::invoke(comp, std::invoke(proj, *sift), std::invoke(proj, *sift_1))) {
            auto val = std::move(*sift);
            do {
                *sift-- = std::move(*sift_1);
            } while (sift != begin &&
                     std::invoke(comp, std::invoke(proj, val), std::invoke(proj, *--sift_1)));
            *sift = std::move(val);
        }
    }
}

/**
 * @brief Unguarded insertion sort assuming a sentinel element strictly less than or equal
 * to any element in [begin, end) exists before begin.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void pdq_unguarded_insertion_sort(I begin, I end, Comp& comp, Proj& proj) {
    if (begin == end) {
        return;
    }
    for (I cur = begin + 1; cur != end; ++cur) {
        I sift = cur;
        I sift_1 = cur - 1;
        if (std::invoke(comp, std::invoke(proj, *sift), std::invoke(proj, *sift_1))) {
            auto val = std::move(*sift);
            do {
                *sift-- = std::move(*sift_1);
            } while (std::invoke(comp, std::invoke(proj, val), std::invoke(proj, *--sift_1)));
            *sift = std::move(val);
        }
    }
}

/**
 * @brief Attempts insertion sort, aborting if more than limit elements are moved.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr bool pdq_partial_insertion_sort(I begin, I end, Comp& comp, Proj& proj,
                                          std::size_t limit = 8) {
    if (begin == end) {
        return true;
    }
    std::size_t moved = 0;
    for (I cur = begin + 1; cur != end; ++cur) {
        I sift = cur;
        I sift_1 = cur - 1;
        if (std::invoke(comp, std::invoke(proj, *sift), std::invoke(proj, *sift_1))) {
            auto val = std::move(*sift);
            do {
                *sift-- = std::move(*sift_1);
            } while (sift != begin &&
                     std::invoke(comp, std::invoke(proj, val), std::invoke(proj, *--sift_1)));
            *sift = std::move(val);
            moved += static_cast<std::size_t>(cur - sift);
        }
        if (moved > limit) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Swaps elements indexed by offset buffers across left and right partitions.
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

enum class ScanDirection { Left, Right };

/**
 * @brief Scans a partition block branchlessly, recording indices of mispositioned elements.
 */
template <ScanDirection Dir, std::size_t Unroll = 8, typename Iter, typename Comp, typename Value>
constexpr void scan_partition_block(Iter& it, std::uint8_t* offsets, std::size_t& num,
                                    std::size_t count, Comp& comp, const Value& pivot) {
    std::size_t i = 0;
    auto step = [&]() {
        if constexpr (Dir == ScanDirection::Left) {
            offsets[num] = static_cast<std::uint8_t>(i++);
            num += !std::invoke(comp, *it, pivot);
            ++it;
        } else {
            offsets[num] = static_cast<std::uint8_t>(++i);
            num += std::invoke(comp, *--it, pivot);
        }
    };

    while (i + Unroll <= count) {
        for (std::size_t u = 0; u < Unroll; ++u) {
            step();
        }
    }
    while (i < count) {
        step();
    }
}

/**
 * @brief BlockQuicksort branchless partitioning algorithm.
 */
template <std::size_t BlockSize = 128, std::random_access_iterator I, typename Comp>
constexpr std::pair<I, bool> pdq_partition_right_branchless(I begin, I end, Comp& comp) {
    using T = std::iter_value_t<I>;
    T pivot = std::move(*begin);
    I first = begin;
    I last = end;

    while (std::invoke(comp, *++first, pivot)) {
    }

    if (first - 1 == begin) {
        while (first < last && !std::invoke(comp, *--last, pivot)) {
        }
    } else {
        while (!std::invoke(comp, *--last, pivot)) {
        }
    }

    bool already_partitioned = first >= last;
    if (!already_partitioned) {
        std::ranges::iter_swap(first, last);
        ++first;

        alignas(64) std::uint8_t offsets_l[BlockSize];
        alignas(64) std::uint8_t offsets_r[BlockSize];

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

            if (left_split > 0) {
                scan_partition_block<ScanDirection::Left>(
                    first, offsets_l, num_l, (left_split >= BlockSize ? BlockSize : left_split),
                    comp, pivot);
            }

            if (right_split > 0) {
                scan_partition_block<ScanDirection::Right>(
                    last, offsets_r, num_r, (right_split >= BlockSize ? BlockSize : right_split),
                    comp, pivot);
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
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr std::pair<I, bool> pdq_partition_right(I begin, I end, Comp& comp, Proj& proj) {
    using T = std::iter_value_t<I>;
    T pivot = std::move(*begin);
    I first = begin;
    I last = end;

    while (std::invoke(comp, std::invoke(proj, *++first), std::invoke(proj, pivot))) {
    }

    if (first - 1 == begin) {
        while (first < last &&
               !std::invoke(comp, std::invoke(proj, *--last), std::invoke(proj, pivot))) {
        }
    } else {
        while (!std::invoke(comp, std::invoke(proj, *--last), std::invoke(proj, pivot))) {
        }
    }

    bool already_partitioned = first >= last;

    while (first < last) {
        std::ranges::iter_swap(first, last);
        while (std::invoke(comp, std::invoke(proj, *++first), std::invoke(proj, pivot))) {
        }
        while (!std::invoke(comp, std::invoke(proj, *--last), std::invoke(proj, pivot))) {
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
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr I pdq_partition_left(I begin, I end, Comp& comp, Proj& proj) {
    using T = std::iter_value_t<I>;
    T pivot = std::move(*begin);
    I first = begin;
    I last = end;

    while (std::invoke(comp, std::invoke(proj, pivot), std::invoke(proj, *--last))) {
    }

    if (last + 1 == end) {
        while (first < last &&
               !std::invoke(comp, std::invoke(proj, pivot), std::invoke(proj, *++first))) {
        }
    } else {
        while (!std::invoke(comp, std::invoke(proj, pivot), std::invoke(proj, *++first))) {
        }
    }

    while (first < last) {
        std::ranges::iter_swap(first, last);
        while (std::invoke(comp, std::invoke(proj, pivot), std::invoke(proj, *--last))) {
        }
        while (!std::invoke(comp, std::invoke(proj, pivot), std::invoke(proj, *++first))) {
        }
    }

    I pivot_pos = last;
    *begin = std::move(*pivot_pos);
    *pivot_pos = std::move(pivot);

    return pivot_pos;
}

} // namespace algoat::detail::pdq

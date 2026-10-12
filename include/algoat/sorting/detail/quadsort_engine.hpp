/**
 * @file quadsort_engine.hpp
 * @brief High-performance Quadsort engine primitives for branchless sorting and merging.
 *
 * Implements Scandum's Quadsort building blocks: tiny sorting networks, branchless
 * bidirectional parity merges, 4-way ping-pong merge networks, and presortedness analysis.
 */

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

namespace algoat::sorting::detail {

/**
 * @brief Swaps two elements if they are out of order according to the comparator and projection.
 */
template <typename I, typename Comp, typename Proj>
constexpr void quadsort_sort2(I a, I b, const Comp& comp, const Proj& proj) {
    if (std::invoke(comp, std::invoke(proj, *b), std::invoke(proj, *a))) {
        std::ranges::iter_swap(a, b);
    }
}

/**
 * @brief Tiny sorting routine for ranges of size 0 to 7 (strictly stable).
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void tiny_sort(I array, std::size_t count, const Comp& comp, const Proj& proj) {
    if (count <= 1) {
        return;
    }
    if (count == 2) {
        if (std::invoke(comp, std::invoke(proj, *(array + 1)), std::invoke(proj, *array))) {
            std::ranges::iter_swap(array, array + 1);
        }
        return;
    }
    for (std::size_t i = 1; i < count; ++i) {
        for (std::size_t j = i; j > 0; --j) {
            if (std::invoke(comp, std::invoke(proj, array[j]), std::invoke(proj, array[j - 1]))) {
                std::ranges::iter_swap(array + j, array + j - 1);
            } else {
                break;
            }
        }
    }
}

/**
 * @brief Bidirectional branchless parity merge for two sorted runs.
 *
 * Merges [from, from + left) and [from + left, from + left + right) into dest.
 * Stable: preserves FIFO order for equal keys from the left and right runs.
 */
template <std::random_access_iterator In, std::random_access_iterator Out, typename Comp,
          typename Proj>
void parity_merge(Out dest, In from, std::size_t left, std::size_t right, const Comp& comp,
                  const Proj& proj) {
    using T = std::iter_value_t<In>;

    if (left == 0 || right == 0) {
        if (left > 0) {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(std::to_address(dest), std::to_address(from), left * sizeof(T));
            } else {
                std::ranges::move(from, from + left, dest);
            }
        } else if (right > 0) {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(std::to_address(dest), std::to_address(from + left), right * sizeof(T));
            } else {
                std::ranges::move(from + left, from + left + right, dest);
            }
        }
        return;
    }

    In left_head = from;
    In right_head = from + left;
    In left_tail = from + left - 1;
    In right_tail = from + left + right - 1;

    Out dest_head = dest;
    Out dest_tail = dest + left + right - 1;

    while (left_head <= left_tail && right_head <= right_tail) {
        // Head merge: left run element preferred when keys are equal (stable)
        if (std::invoke(comp, std::invoke(proj, *right_head), std::invoke(proj, *left_head))) {
            *dest_head++ = std::ranges::iter_move(right_head++);
        } else {
            *dest_head++ = std::ranges::iter_move(left_head++);
        }

        if (left_head > left_tail || right_head > right_tail) {
            break;
        }

        // Tail merge: right run element preferred when keys are equal (stable)
        if (std::invoke(comp, std::invoke(proj, *right_tail), std::invoke(proj, *left_tail))) {
            *dest_tail-- = std::ranges::iter_move(left_tail--);
        } else {
            *dest_tail-- = std::ranges::iter_move(right_tail--);
        }
    }

    if constexpr (std::is_trivially_copyable_v<T>) {
        if (left_head <= left_tail) {
            std::size_t rem = static_cast<std::size_t>(left_tail - left_head + 1);
            std::memcpy(std::to_address(dest_head), std::to_address(left_head), rem * sizeof(T));
        }
        if (right_head <= right_tail) {
            std::size_t rem = static_cast<std::size_t>(right_tail - right_head + 1);
            std::memcpy(std::to_address(dest_head), std::to_address(right_head), rem * sizeof(T));
        }
    } else {
        while (left_head <= left_tail) {
            *dest_head++ = std::ranges::iter_move(left_head++);
        }
        while (right_head <= right_tail) {
            *dest_head++ = std::ranges::iter_move(right_head++);
        }
    }
}

/**
 * @brief Quadsort 4-way ping-pong merge network.
 *
 * Merges four sorted sub-runs of length quad1, quad2, quad3, and quad4 simultaneously.
 * Pass 1: Merges (run1, run2) and (run3, run4) from array into swap.
 * Pass 2: Merges the two combined runs from swap back into array.
 * Halves memory write passes compared to standard cascading mergesort.
 */
template <std::random_access_iterator I, typename SwapBuf, typename Comp, typename Proj>
void quad_merge_4way(I array, SwapBuf& swap, std::size_t quad1, std::size_t quad2,
                     std::size_t quad3, std::size_t quad4, const Comp& comp, const Proj& proj) {
    std::size_t half1 = quad1 + quad2;
    std::size_t half2 = quad3 + quad4;

    I run1 = array;
    I run3 = array + half1;

    // Check if runs are already mutually in order
    bool pair1_ordered =
        !std::invoke(comp, std::invoke(proj, array[quad1]), std::invoke(proj, array[quad1 - 1]));
    bool pair2_ordered = !std::invoke(comp, std::invoke(proj, array[half1 + quad3]),
                                      std::invoke(proj, array[half1 + quad3 - 1]));

    if (pair1_ordered && pair2_ordered &&
        !std::invoke(comp, std::invoke(proj, array[half1]), std::invoke(proj, array[half1 - 1]))) {
        // Entire 4-way block is already sorted
        return;
    }

    parity_merge(swap.data(), run1, quad1, quad2, comp, proj);
    parity_merge(swap.data() + half1, run3, quad3, quad4, comp, proj);
    parity_merge(array, swap.data(), half1, half2, comp, proj);
}

/**
 * @brief Tail swap helper for partitions of size <= 96 using 4-way merge networks.
 */
template <std::random_access_iterator I, typename SwapBuf, typename Comp, typename Proj>
void tail_swap(I array, SwapBuf& swap, std::size_t count, const Comp& comp, const Proj& proj) {
    if (count < 8) {
        tiny_sort(array, count, comp, proj);
        return;
    }

    std::size_t half1 = count / 2;
    std::size_t quad1 = half1 / 2;
    std::size_t quad2 = half1 - quad1;
    std::size_t half2 = count - half1;
    std::size_t quad3 = half2 / 2;
    std::size_t quad4 = half2 - quad3;

    I cursor = array;
    tail_swap(cursor, swap, quad1, comp, proj);
    cursor += quad1;
    tail_swap(cursor, swap, quad2, comp, proj);
    cursor += quad2;
    tail_swap(cursor, swap, quad3, comp, proj);
    cursor += quad3;
    tail_swap(cursor, swap, quad4, comp, proj);

    quad_merge_4way(array, swap, quad1, quad2, quad3, quad4, comp, proj);
}

/**
 * @brief Computes the median of three elements without conditional branching.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr I median_of_three(I array, std::size_t v0, std::size_t v1, std::size_t v2,
                            const Comp& comp, const Proj& proj) {
    bool x = std::invoke(comp, std::invoke(proj, array[v1]), std::invoke(proj, array[v0]));
    bool y = std::invoke(comp, std::invoke(proj, array[v2]), std::invoke(proj, array[v0]));
    bool z = std::invoke(comp, std::invoke(proj, array[v2]), std::invoke(proj, array[v1]));

    std::array<std::size_t, 3> v = {v0, v1, v2};
    return array + v[(x == y) + (y ^ z)];
}

/**
 * @brief Branchless ninther (pseudomedian of 9 elements).
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr I median_of_nine(I array, std::size_t count, const Comp& comp, const Proj& proj) {
    std::size_t div = count / 16;
    if (div == 0) {
        div = 1;
    }

    I x = median_of_three(array, div * 2, div * 1, div * 4, comp, proj);
    I y = median_of_three(array, div * 8, div * 6, div * 10, comp, proj);
    I z = median_of_three(array, div * 14, div * 12, div * 15, comp, proj);

    std::size_t ix = static_cast<std::size_t>(x - array);
    std::size_t iy = static_cast<std::size_t>(y - array);
    std::size_t iz = static_cast<std::size_t>(z - array);

    return median_of_three(array, ix, iy, iz, comp, proj);
}

/**
 * @brief Fast binary median search between two sorted runs of length @p len.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
I binary_median(I pta, I ptb, std::size_t len, const Comp& comp, const Proj& proj) {
    while (len /= 2) {
        if (!std::invoke(comp, std::invoke(proj, *(ptb + len)), std::invoke(proj, *(pta + len)))) {
            pta += len;
        } else {
            ptb += len;
        }
    }
    return std::invoke(comp, std::invoke(proj, *ptb), std::invoke(proj, *pta)) ? pta : ptb;
}

enum class PresortedResult { AlreadySorted, ReverseSorted, NeedsPartitioning };

/**
 * @brief Shared 4-quadrant presortedness analyzer.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
PresortedResult analyze_presortedness(I array, std::size_t count, const Comp& comp,
                                      const Proj& proj) {
    std::size_t half1 = count / 2;
    std::size_t quad1 = half1 / 2;
    std::size_t quad2 = half1 - quad1;
    std::size_t half2 = count - half1;
    std::size_t quad3 = half2 / 2;
    std::size_t quad4 = half2 - quad3;

    I pta = array;
    I ptb = array + quad1;
    I ptc = array + half1;
    I ptd = array + half1 + quad3;

    std::size_t min_quad = std::min({quad1, quad2, quad3, quad4});
    if (min_quad <= 1) {
        return PresortedResult::NeedsPartitioning;
    }
    std::size_t check_len = std::min(min_quad - 1, std::size_t{32});
    std::size_t inversions = 0;

    for (std::size_t i = 0; i < check_len; ++i) {
        inversions += std::invoke(comp, std::invoke(proj, pta[i + 1]), std::invoke(proj, pta[i]));
        inversions += std::invoke(comp, std::invoke(proj, ptb[i + 1]), std::invoke(proj, ptb[i]));
        inversions += std::invoke(comp, std::invoke(proj, ptc[i + 1]), std::invoke(proj, ptc[i]));
        inversions += std::invoke(comp, std::invoke(proj, ptd[i + 1]), std::invoke(proj, ptd[i]));
    }

    if (inversions == 0) {
        if (!std::invoke(comp, std::invoke(proj, array[quad1]),
                         std::invoke(proj, array[quad1 - 1])) &&
            !std::invoke(comp, std::invoke(proj, array[half1]),
                         std::invoke(proj, array[half1 - 1])) &&
            !std::invoke(comp, std::invoke(proj, array[half1 + quad3]),
                         std::invoke(proj, array[half1 + quad3 - 1]))) {
            bool all_sorted = true;
            for (std::size_t i = 1; i < count; ++i) {
                if (std::invoke(comp, std::invoke(proj, array[i]),
                                std::invoke(proj, array[i - 1]))) {
                    all_sorted = false;
                    break;
                }
            }
            if (all_sorted) {
                return PresortedResult::AlreadySorted;
            }
        }
    }

    if (inversions == check_len * 4) {
        bool all_reversed = true;
        for (std::size_t i = 1; i < count; ++i) {
            if (std::invoke(comp, std::invoke(proj, array[i - 1]), std::invoke(proj, array[i]))) {
                all_reversed = false;
                break;
            }
        }
        if (all_reversed) {
            return PresortedResult::ReverseSorted;
        }
    }

    return PresortedResult::NeedsPartitioning;
}

} // namespace algoat::sorting::detail

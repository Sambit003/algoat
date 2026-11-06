/**
 * @file smoothsort.hpp
 * @brief Dijkstra's Smoothsort: In-place adaptive heapsort with O(1) auxiliary space.
 *
 * Implements Dijkstra's Smoothsort (EWD796a / Science of Computer Programming 1982)
 * using an implicit forest of Leonardo heaps.
 *
 * Characteristics:
 * - Best-case Time: Strictly linear O(N) comparisons on presorted data (with 0 swaps)
 * - Worst-case Time: Guaranteed O(N log N) comparisons (bounded by O(N log N))
 * - Space Complexity: Strictly O(1) auxiliary memory (zero dynamic allocations)
 * - Stability: Unstable comparison sort
 */

#pragma once

#include "algoat/core/registry.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

namespace detail::smooth {

/**
 * @brief Advances Leonardo scale pair (b, c) upward to next Leonardo number.
 *
 * b_{k+1} = b_k + c_k + 1, c_{k+1} = b_k
 */
constexpr void up(std::size_t& b, std::size_t& c) noexcept {
    const std::size_t next_b = b + c + 1;
    c = b;
    b = next_b;
}

/**
 * @brief Steps Leonardo scale pair (b, c) downward to previous Leonardo number.
 *
 * b_{k-1} = c_k, c_{k-1} = b_k - c_k - 1
 */
constexpr void down(std::size_t& b, std::size_t& c) noexcept {
    const std::size_t prev_c = b - c - 1;
    b = c;
    c = prev_c;
}

/**
 * @brief Sifts down root element within a single Leonardo tree of order parameters (b1, c1).
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void sift(I first, std::size_t r1, std::size_t b1, std::size_t c1, Comp& comp,
                    Proj& proj) {
    auto compare_indices = [&](std::size_t i, std::size_t j) -> bool {
        return bool(
            std::invoke(comp, std::invoke(proj, *(first + i)), std::invoke(proj, *(first + j))));
    };

    while (b1 >= 3) {
        std::size_t r2 = r1 - b1 + c1; // Left child root
        if (compare_indices(r2, r1 - 1)) {
            r2 = r1 - 1; // Right child is greater
            down(b1, c1);
        }
        if (!compare_indices(r1, r2)) {
            b1 = 1; // Father >= largest child: stop
        } else {
            std::ranges::iter_swap(first + r1, first + r2);
            r1 = r2;
            down(b1, c1);
        }
    }
}

/**
 * @brief Trinkle: Restores forest root ascending ordering and cascades down into trees.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void trinkle(I first, std::size_t r1, std::uint64_t p_val, std::size_t b_val,
                       std::size_t c_val, Comp& comp, Proj& proj) {
    auto compare_indices = [&](std::size_t i, std::size_t j) -> bool {
        return bool(
            std::invoke(comp, std::invoke(proj, *(first + i)), std::invoke(proj, *(first + j))));
    };

    std::uint64_t p1 = p_val;
    std::size_t b1 = b_val;
    std::size_t c1 = c_val;

    while (p1 > 0) {
        while ((p1 & 1ULL) == 0ULL) {
            p1 >>= 1;
            up(b1, c1);
        }
        const std::size_t r3 = r1 - b1; // Stepson root
        if (p1 == 1 || !compare_indices(r1, r3)) {
            p1 = 0; // Forest invariant satisfied
        } else {
            --p1;
            if (b1 == 1) {
                std::ranges::iter_swap(first + r1, first + r3);
                r1 = r3;
            } else if (b1 >= 3) {
                std::size_t r2 = r1 - b1 + c1; // Left child root
                if (compare_indices(r2, r1 - 1)) {
                    r2 = r1 - 1; // Right child is greater
                    down(b1, c1);
                    p1 <<= 1;
                }
                if (!compare_indices(r3, r2)) {
                    std::ranges::iter_swap(first + r1, first + r3);
                    r1 = r3;
                } else {
                    std::ranges::iter_swap(first + r1, first + r2);
                    r1 = r2;
                    down(b1, c1);
                    p1 = 0;
                }
            }
        }
    }
    sift(first, r1, b1, c1, comp, proj);
}

/**
 * @brief Semitrinkle: Fast-path root order check during Leonardo tree deconstruction.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void semitrinkle(I first, std::size_t r, std::uint64_t p_val, std::size_t b_val,
                           std::size_t c_val, Comp& comp, Proj& proj) {
    const std::size_t r1 = r - c_val;
    if (bool(
            std::invoke(comp, std::invoke(proj, *(first + r)), std::invoke(proj, *(first + r1))))) {
        std::ranges::iter_swap(first + r, first + r1);
        trinkle(first, r1, p_val, b_val, c_val, comp, proj);
    }
}

} // namespace detail::smooth

/**
 * @brief Sorts the iterator range [first, last) in-place using Dijkstra's Smoothsort.
 *
 * Smoothsort maintains a forest of heaps whose subtree sizes correspond to Leonardo numbers.
 * Because the roots of the Leonardo heaps are kept in ascending order, presorted sequences
 * complete in strictly O(N) comparisons and zero swaps, while worst-case sequences require
 * guaranteed O(N log N) comparisons in strictly O(1) auxiliary space.
 *
 * @tparam I Random-access iterator type.
 * @tparam S Sentinel type.
 * @tparam Comp Strict weak ordering comparator.
 * @tparam Proj Projection callable.
 * @param first Beginning of range.
 * @param last Sentinel / end of range.
 * @param comp Strict weak ordering comparator (default: std::ranges::less).
 * @param proj Element projection callable (default: std::identity).
 * @return Iterator pointing to the end of the range.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
constexpr I smoothsort(I first, S last, Comp comp = {}, Proj proj = {}) {
    const auto n_diff = std::ranges::distance(first, last);
    if (n_diff <= 1) {
        return first + n_diff;
    }

    const std::size_t N = static_cast<std::size_t>(n_diff);

    // Fast-path: strictly descending sequence reversal to guarantee <= 1.5 * N * log2(N)
    std::size_t desc_count = 1;
    while (desc_count < N && bool(std::invoke(comp, std::invoke(proj, *(first + desc_count)),
                                              std::invoke(proj, *(first + (desc_count - 1)))))) {
        ++desc_count;
    }
    if (desc_count == N) {
        std::ranges::reverse(first, last);
        return first + n_diff;
    }

    // Phase 1: Accumulation (building forest of Leonardo heaps)
    std::size_t q = 1;
    std::size_t r = 0;
    std::uint64_t p = 1ULL;
    std::size_t b = 1;
    std::size_t c = 1;

    while (q != N) {
        const std::size_t r1 = r;
        if ((p & 7ULL) == 3ULL) {
            detail::smooth::sift(first, r1, b, c, comp, proj);
            p = (p + 1ULL) >> 2;
            detail::smooth::up(b, c);
            detail::smooth::up(b, c);
        } else if ((p & 3ULL) == 1ULL) {
            if (q + c < N) {
                detail::smooth::sift(first, r1, b, c, comp, proj);
            } else {
                detail::smooth::trinkle(first, r1, p, b, c, comp, proj);
            }
            detail::smooth::down(b, c);
            p <<= 1;
            while (b != 1) {
                detail::smooth::down(b, c);
                p <<= 1;
            }
            p += 1ULL;
        }
        ++q;
        ++r;
    }
    detail::smooth::trinkle(first, r, p, b, c, comp, proj);

    // Phase 2: Reduction / Extraction (deconstructing forest from right to left)
    while (q != 1) {
        --q;
        if (b == 1) {
            --r;
            --p;
            while ((p & 1ULL) == 0ULL) {
                p >>= 1;
                detail::smooth::up(b, c);
            }
        } else if (b >= 3) {
            --p;
            r = r - b + c; // Left child root
            if (p > 0) {
                detail::smooth::semitrinkle(first, r, p, b, c, comp, proj);
            }
            detail::smooth::down(b, c);
            p = (p << 1) + 1ULL;
            r = r + c; // Right child root
            detail::smooth::semitrinkle(first, r, p, b, c, comp, proj);
            detail::smooth::down(b, c);
            p = (p << 1) + 1ULL;
        }
    }

    return first + n_diff;
}

/**
 * @brief Sorts a random-access range in-place using Dijkstra's Smoothsort.
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator.
 * @tparam Proj Projection callable.
 * @param data Range to sort.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Borrowed iterator pointing to the end of the range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> smoothsort(R&& data, Comp comp = {}, Proj proj = {}) {
    return smoothsort(std::ranges::begin(data), std::ranges::end(data), std::move(comp),
                      std::move(proj));
}

/**
 * @brief Sorts a contiguous span in-place using Dijkstra's Smoothsort with a custom comparator.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator.
 * @param data Span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare = std::ranges::less>
void smoothsort(std::span<T> data, Compare comp = {}) {
    smoothsort(data.begin(), data.end(), std::move(comp), std::identity{});
}

/**
 * @struct SmoothSort
 * @brief Zero-overhead adapter struct for Smoothsort satisfying SortAlgorithm concept.
 */
struct SmoothSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "smoothsort";
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    template <typename T> void sort(std::span<T> data) const {
        smoothsort(data);
    }

    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        smoothsort(data, std::move(comp));
    }
};

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "smoothsort", ::algoat::sorting::SmoothSort)

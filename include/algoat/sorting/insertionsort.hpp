/**
 * @file insertionsort.hpp
 * @brief In-place comparison-based Insertion Sort algorithm.
 */

#pragma once

#include "algoat/core/traits.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

/**
 * @brief Sorts a C++20 random-access range using insertion sort with custom comparator and
 * projection.
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param data Range of elements to sort in-place.
 * @param comp Comparator callable.
 * @param proj Projection callable.
 * @return Iterator pointing to the end of the range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> insertionsort(R&& data, Comp comp = {},
                                                            Proj proj = {}) {
    auto comp_proj = ::algoat::detail::make_comp_proj(comp, proj);
    for (auto i = std::ranges::begin(data); i != std::ranges::end(data); ++i) {
        for (auto j = i; j != std::ranges::begin(data) && comp_proj(*j, *(j - 1)); --j) {
            std::ranges::iter_swap(j, j - 1);
        }
    }
    return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
}

/**
 * @brief Sorts the given span using insertion sort with a custom comparator.
 * @tparam T Element type supporting move construction/assignment.
 * @tparam Compare Callable comparator defining strict weak ordering.
 *
 * @param data Span of elements to sort in-place.
 * @param comp Comparator callable.
 */
template <typename T, typename Compare> void insertionsort(std::span<T> data, Compare comp) {
    const std::size_t n = data.size();
    for (std::size_t i = 1; i < n; ++i) {
        T key = std::move(data[i]);
        std::size_t j = i;
        while (j > 0 && comp(key, data[j - 1])) {
            data[j] = std::move(data[j - 1]);
            --j;
        }
        data[j] = std::move(key);
    }
}

/**
 * @brief Sorts the given span using insertion sort.
 * @tparam T Element type supporting @c operator< and move construction/assignment.
 *
 * @param data Span of elements to sort in-place.
 */
template <typename T> void insertionsort(std::span<T> data) {
    insertionsort(data, std::less<T>{});
}

/**
 * @struct InsertionSort
 * @brief Standard stable Insertion Sort implementation with move semantics.
 *
 * Efficient for small sequences (@c N < 32) and nearly sorted data. Used as
 * the default base-case sort in hybrid algorithms (IntroSort, TimSort, BlockSort).
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Insertion.
 * - <b>Stability:</b> Stable (preserves relative order of equal keys).
 *
 * @par Time Complexity:
 * - Best Case: @c O(N) (already sorted)
 * - Average Case: @c O(N^2)
 * - Worst Case: @c O(N^2) (reverse sorted)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space (in-place)
 */
struct InsertionSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "insertionsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "insertionsort";
    }

    /**
     * @brief Sorts the given span using insertion sort.
     * @tparam T Element type supporting @c operator< and move construction/assignment.
     *
     * @param data Span of elements to sort in-place.
     */
    template <typename T> void sort(std::span<T> data) const {
        insertionsort(data);
    }

    /**
     * @brief Sorts the given span using insertion sort with a custom comparator.
     * @tparam T Element type supporting move construction/assignment.
     * @tparam Compare Strict weak ordering comparator callable.
     *
     * @param data Span of elements to sort in-place.
     * @param comp Comparator callable.
     */
    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        insertionsort(data, comp);
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0 (effective from size 0 upwards).
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "insertionsort", ::algoat::sorting::InsertionSort)

/**
 * @file ips4o.hpp
 * @brief In-Place Parallel Super Scalar Samplesort (IPS⁴o) implementation.
 *
 * Implements Axtmann, Witt, Ferizovic, and Sanders' (ESA 2017) In-Place Parallel
 * Super Scalar Samplesort algorithm with branchless decision tree classification,
 * in-place block permutations, equality buckets for duplicate handling,
 * and adaptive thread scheduling.
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/core/traits.hpp"
#include "algoat/sorting/detail/ips4o/ips4o.hpp"
#include "algoat/sorting/pdqsort.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace algoat::sorting {

/**
 * @brief Sorts a random-access range [first, last) in-place using In-Place Parallel Super Scalar
 * Samplesort.
 *
 * @tparam I Random-access iterator type satisfying std::random_access_iterator.
 * @tparam S Sentinel type satisfying std::sentinel_for<I>.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Element projection callable.
 *
 * @param first Beginning of the range to sort.
 * @param last Sentinel end of the range to sort.
 * @param comp Strict weak ordering comparator (default: std::ranges::less).
 * @param proj Element projection callable (default: std::identity).
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
void ips4o_parallel_sort(I first, S last, Comp comp, Proj proj, int num_threads) {
    const auto n = std::ranges::distance(first, last);
    if (n <= 1)
        return;

    using ValueType = std::iter_value_t<I>;

    if constexpr (!std::is_copy_constructible_v<ValueType>) {
        ::algoat::sorting::pdqsort(first, last, std::move(comp), std::move(proj));
    } else {
        auto comp_proj = ::algoat::detail::make_comp_proj(std::move(comp), std::move(proj));
        ::algoat::sorting::detail::ips4o::parallel::sort(first, first + n, std::move(comp_proj),
                                                         num_threads);
    }
}

template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
void ips4o_parallel_sort(I first, S last, Comp comp = {}, Proj proj = {}) {
    const auto n = std::ranges::distance(first, last);
    if (n <= 1)
        return;

    using ValueType = std::iter_value_t<I>;

    if constexpr (!std::is_copy_constructible_v<ValueType>) {
        ::algoat::sorting::pdqsort(first, last, std::move(comp), std::move(proj));
    } else {
        auto comp_proj = ::algoat::detail::make_comp_proj(std::move(comp), std::move(proj));
        ::algoat::sorting::detail::ips4o::parallel::sort(first, first + n, std::move(comp_proj));
    }
}

/**
 * @brief Sorts a C++20 random-access range in-place using In-Place Parallel Super Scalar
 * Samplesort.
 *
 * @tparam R Random-access range type satisfying std::ranges::random_access_range.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Element projection callable.
 *
 * @param r Range to sort.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Iterator to the end of the range, or std::ranges::dangling if an rvalue non-borrowed
 * range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
void ips4o_parallel_sort(R&& r, Comp comp = {}, Proj proj = {}) {
    ips4o_parallel_sort(std::ranges::begin(r), std::ranges::end(r), std::move(comp),
                        std::move(proj));
}

/**
 * @brief Overload sorting a contiguous std::span in-place.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param data Contiguous span of elements.
 * @param comp Comparator callable.
 */
template <typename T, typename Compare = std::ranges::less>
void ips4o_parallel_sort(std::span<T> data, Compare comp = {}) {
    ips4o_parallel_sort(data.begin(), data.end(), std::move(comp), std::identity{});
}

/**
 * @brief Convenience aliases providing standard library-style ips4o naming.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
void ips4o(I first, S last, Comp comp = {}, Proj proj = {}) {
    ips4o_parallel_sort(first, last, std::move(comp), std::move(proj));
}

template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
void ips4o(R&& r, Comp comp = {}, Proj proj = {}) {
    ips4o_parallel_sort(std::forward<R>(r), std::move(comp), std::move(proj));
}

template <typename T, typename Compare = std::ranges::less>
void ips4o(std::span<T> data, Compare comp = {}) {
    ips4o_parallel_sort(data, std::move(comp));
}

/**
 * @struct Ips4oSort
 * @brief Zero-overhead adapter struct for In-Place Parallel Super Scalar Samplesort (IPS⁴o).
 *
 * Conforms to the algoat::sorting::SortAlgorithm concept for static polymorphism
 * and dynamic registry integration.
 */
struct Ips4oSort {
    /**
     * @brief Unique algorithm name identifier.
     * @return "ips4o"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "ips4o";
    }

    /**
     * @brief Preferred minimum size threshold for parallel dispatch.
     * @return 65536
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 65536;
    }

    /**
     * @brief Sorts the span in-place using default comparator.
     * @tparam T Element type.
     * @param data Contiguous span of elements.
     */
    template <typename T> void sort(std::span<T> data) const {
        ips4o_parallel_sort(data);
    }

    /**
     * @brief Sorts the span in-place using a custom comparator.
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator.
     * @param data Contiguous span of elements.
     * @param comp Comparator callable.
     */
    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        ips4o_parallel_sort(data, std::move(comp));
    }
};

using IPS4OSort = Ips4oSort;

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "ips4o", ::algoat::sorting::Ips4oSort)

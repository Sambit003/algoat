/**
 * @file mergesort.hpp
 * @brief Stable Merge Sort implementation using auxiliary buffer.
 */

#pragma once

#include "algoat/core/scratch_buffer.hpp"
#include "algoat/core/traits.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory_resource>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace algoat::sorting {

namespace detail {

template <typename RandIt, typename T, typename Compare>
void merge_impl(RandIt arr, T* temp, std::size_t left, std::size_t mid, std::size_t right,
                Compare& comp) {
    using Diff = std::iter_difference_t<RandIt>;
    std::size_t i = left;
    std::size_t j = mid + 1;
    std::size_t k = left;

    while (i <= mid && j <= right) {
        auto&& lhs = arr[static_cast<Diff>(i)];
        auto&& rhs = arr[static_cast<Diff>(j)];
        if (!comp(rhs, lhs)) { // <= ensures stability
            temp[k++] = std::move(lhs);
            ++i;
        } else {
            temp[k++] = std::move(rhs);
            ++j;
        }
    }

    while (i <= mid) {
        temp[k++] = std::move(arr[static_cast<Diff>(i++)]);
    }

    while (j <= right) {
        temp[k++] = std::move(arr[static_cast<Diff>(j++)]);
    }

    for (std::size_t p = left; p <= right; ++p) {
        arr[static_cast<Diff>(p)] = std::move(temp[p]);
    }
}

template <typename RandIt, typename T, typename Compare>
void mergesort_impl(RandIt arr, T* temp, std::size_t left, std::size_t right, Compare& comp) {
    if (left < right) {
        std::size_t mid = left + (right - left) / 2;
        mergesort_impl(arr, temp, left, mid, comp);
        mergesort_impl(arr, temp, mid + 1, right, comp);
        merge_impl(arr, temp, left, mid, right, comp);
    }
}

} // namespace detail

/**
 * @brief Sorts the span in-place using top-down merge sort with a custom comparator and optional
 * memory resource.
 * @tparam T Element type supporting move operations.
 * @tparam Extent Static or dynamic span extent.
 * @tparam Compare Strict weak ordering callable.
 *
 * @param data Span of elements to sort.
 * @param comp Custom comparator callable.
 * @param mr Pointer to a polymorphic memory resource for the scratchpad buffer.
 */
template <typename T, std::size_t Extent, typename Compare>
    requires(!std::is_convertible_v<Compare, std::pmr::memory_resource*>)
void mergesort(std::span<T, Extent> data, Compare comp,
               std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    if (data.size() <= 1) {
        return;
    }
    algoat::core::ScratchBuffer<T> buffer(data.size(), mr);
    detail::mergesort_impl(data.data(), buffer.data(), 0, data.size() - 1, comp);
}

/**
 * @brief Sorts the span in-place using top-down merge sort with an optional memory resource.
 * @tparam T Element type supporting @c operator< and move operations.
 * @tparam Extent Static or dynamic span extent.
 *
 * @param data Span of elements to sort.
 * @param mr Pointer to a polymorphic memory resource for the scratchpad buffer.
 */
template <typename T, std::size_t Extent>
void mergesort(std::span<T, Extent> data,
               std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    mergesort(data, std::less<>{}, mr);
}

/**
 * @brief Sorts a C++20 random-access range in-place using top-down merge sort with custom
 * comparator, projection, and optional memory resource.
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param data Range of elements to sort in-place.
 * @param comp Custom comparator callable.
 * @param proj Element projection callable.
 * @param mr Pointer to a polymorphic memory resource for the scratchpad buffer.
 * @return Iterator pointing to the end of the range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires(!::algoat::detail::is_std_span_v<R> &&
             !std::is_convertible_v<Comp, std::pmr::memory_resource*> &&
             !std::is_convertible_v<Proj, std::pmr::memory_resource*> &&
             std::sortable<std::ranges::iterator_t<R>, Comp, Proj>)
inline std::ranges::borrowed_iterator_t<R>
mergesort(R&& data, Comp comp = {}, Proj proj = {},
          std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    if constexpr (std::ranges::contiguous_range<R> && std::ranges::sized_range<R>) {
        auto span_data = ::algoat::detail::to_span(data);
        if (span_data.size() <= 1) {
            return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
        }
        auto comp_proj = ::algoat::detail::make_comp_proj(std::move(comp), std::move(proj));
        mergesort(span_data, comp_proj, mr);
        return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
    } else {
        const auto count = static_cast<std::size_t>(std::ranges::distance(data));
        if (count <= 1) {
            return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
        }
        using ValueType = std::ranges::range_value_t<R>;
        auto comp_proj = ::algoat::detail::make_comp_proj(std::move(comp), std::move(proj));
        algoat::core::ScratchBuffer<ValueType> buffer(count, mr);
        detail::mergesort_impl(std::ranges::begin(data), buffer.data(), 0, count - 1, comp_proj);
        return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
    }
}

/**
 * @brief Sorts a C++20 random-access range in-place using top-down merge sort with a custom
 * comparator and memory resource.
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator callable.
 *
 * @param data Range of elements to sort in-place.
 * @param comp Custom comparator callable.
 * @param mr Pointer to a polymorphic memory resource for the scratchpad buffer.
 * @return Iterator pointing to the end of the range.
 */
template <std::ranges::random_access_range R, typename Comp>
    requires(!::algoat::detail::is_std_span_v<R> &&
             !std::is_convertible_v<Comp, std::pmr::memory_resource*> &&
             std::sortable<std::ranges::iterator_t<R>, Comp>)
inline std::ranges::borrowed_iterator_t<R> mergesort(R&& data, Comp comp,
                                                     std::pmr::memory_resource* mr) {
    return mergesort(std::forward<R>(data), std::move(comp), std::identity{}, mr);
}

/**
 * @brief Sorts a C++20 random-access range in-place using top-down merge sort with an explicit
 * memory resource.
 *
 * @tparam R Random-access range type.
 *
 * @param data Range of elements to sort in-place.
 * @param mr Pointer to a polymorphic memory resource for the scratchpad buffer.
 * @return Iterator pointing to the end of the range.
 */
template <std::ranges::random_access_range R>
    requires(!::algoat::detail::is_std_span_v<R> && std::sortable<std::ranges::iterator_t<R>>)
inline std::ranges::borrowed_iterator_t<R> mergesort(R&& data, std::pmr::memory_resource* mr) {
    return mergesort(std::forward<R>(data), std::ranges::less{}, std::identity{}, mr);
}

/**
 * @struct MergeSort
 * @brief Top-down, divide-and-conquer stable sorting algorithm.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Divide & Conquer.
 * - <b>Stability:</b> Stable (preserves order of equivalent keys via @c <= merge comparison).
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N log N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(N) auxiliary buffer space
 */
struct MergeSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "mergesort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "mergesort";
    }

    /**
     * @brief Sorts the span in-place using top-down merge sort.
     * @tparam T Element type supporting @c operator< and move operations.
     * @tparam Extent Static or dynamic span extent.
     *
     * @param data Span of elements to sort.
     */
    template <typename T, std::size_t Extent = std::dynamic_extent>
    void sort(std::span<T, Extent> data) const {
        mergesort(data);
    }

    /**
     * @brief Sorts the span in-place using top-down merge sort with a memory resource.
     * @tparam T Element type supporting @c operator< and move operations.
     * @tparam Extent Static or dynamic span extent.
     *
     * @param data Span of elements to sort.
     * @param mr Pointer to a polymorphic memory resource for the scratchpad buffer.
     */
    template <typename T, std::size_t Extent = std::dynamic_extent>
    void sort(std::span<T, Extent> data, std::pmr::memory_resource* mr) const {
        mergesort(data, mr);
    }

    /**
     * @brief Sorts the span in-place using top-down merge sort with a custom comparator.
     * @tparam T Element type supporting move operations.
     * @tparam Extent Static or dynamic span extent.
     * @tparam Compare Strict weak ordering callable.
     *
     * @param data Span of elements to sort.
     * @param comp Custom comparator callable.
     */
    template <typename T, std::size_t Extent = std::dynamic_extent, typename Compare>
        requires(!std::is_convertible_v<Compare, std::pmr::memory_resource*>)
    void sort(std::span<T, Extent> data, Compare comp) const {
        mergesort(data, std::move(comp));
    }

    /**
     * @brief Sorts the span in-place using top-down merge sort with a custom comparator and
     * memory resource.
     * @tparam T Element type supporting move operations.
     * @tparam Extent Static or dynamic span extent.
     * @tparam Compare Strict weak ordering callable.
     *
     * @param data Span of elements to sort.
     * @param comp Custom comparator callable.
     * @param mr Pointer to a polymorphic memory resource for the scratchpad buffer.
     */
    template <typename T, std::size_t Extent = std::dynamic_extent, typename Compare>
        requires(!std::is_convertible_v<Compare, std::pmr::memory_resource*>)
    void sort(std::span<T, Extent> data, Compare comp, std::pmr::memory_resource* mr) const {
        mergesort(data, std::move(comp), mr);
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 32
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 32;
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "mergesort", ::algoat::sorting::MergeSort)

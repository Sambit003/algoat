/**
 * @file linear_search.hpp
 * @brief Linear Search algorithm implementation.
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::searching {

/**
 * @brief Linearly searches for target in a C++20 input range using a custom binary predicate and
 * projection.
 *
 * @tparam R Input range type.
 * @tparam T Target value type.
 * @tparam Proj Projection callable.
 * @tparam Pred Binary predicate callable testing equivalence.
 *
 * @param range Input range of elements to search.
 * @param target Value to locate.
 * @param pred Binary predicate callable.
 * @param proj Element projection callable.
 * @return Index of the first matching element, or @c std::nullopt.
 */
template <std::ranges::input_range R, typename T = std::ranges::range_value_t<R>,
          typename Proj = std::identity, typename Pred = std::ranges::equal_to>
    requires std::indirect_binary_predicate<Pred, const T*,
                                            std::projected<std::ranges::iterator_t<R>, Proj>>
std::optional<std::size_t> linear_search(R&& range, const T& target, Pred pred = {},
                                         Proj proj = {}) {
    std::size_t idx = 0;
    for (auto it = std::ranges::begin(range); it != std::ranges::end(range); ++it, ++idx) {
        if (std::invoke(pred, target, std::invoke(proj, *it))) {
            return idx;
        }
    }
    return std::nullopt;
}

/**
 * @brief Linearly searches for target in the span using a custom binary predicate.
 * @tparam T Element type.
 * @tparam BinaryPredicate Binary predicate callable testing equivalence.
 *
 * @param data Span of elements to search.
 * @param target Value to locate.
 * @param pred Binary predicate callable.
 * @return Index of the first matching element, or @c std::nullopt.
 */
template <typename T, typename BinaryPredicate>
std::optional<std::size_t> linear_search(std::span<const T> data, const T& target,
                                         BinaryPredicate pred) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (pred(data[i], target)) {
            return i;
        }
    }
    return std::nullopt;
}

/**
 * @brief Linearly searches for target in the span.
 *
 * @tparam T Element type supporting @c operator==.
 * @param data Span of elements to search.
 * @param target Value to locate.
 * @return Index of the first matching element, or @c std::nullopt.
 */
template <typename T>
std::optional<std::size_t> linear_search(std::span<const T> data, const T& target) {
    return linear_search(data, target, std::equal_to<>{});
}

/**
 * @brief Linearly searches for target in a mutable span.
 *
 * @tparam T Element type supporting @c operator==.
 * @param data Mutable span of elements to search.
 * @param target Value to locate.
 * @return Index of the first matching element, or @c std::nullopt.
 */
template <typename T> std::optional<std::size_t> linear_search(std::span<T> data, const T& target) {
    return linear_search(std::span<const T>{data.data(), data.size()}, target);
}

/**
 * @brief Linearly searches for target in a mutable span using a custom binary predicate.
 *
 * @tparam T Element type.
 * @tparam BinaryPredicate Binary predicate callable testing equivalence.
 * @param data Mutable span of elements to search.
 * @param target Value to locate.
 * @param pred Binary predicate callable.
 * @return Index of the first matching element, or @c std::nullopt.
 */
template <typename T, typename BinaryPredicate>
std::optional<std::size_t> linear_search(std::span<T> data, const T& target, BinaryPredicate pred) {
    return linear_search(std::span<const T>{data.data(), data.size()}, target, std::move(pred));
}

/**
 * @struct LinearSearch
 * @brief Sequential search scanning elements from left to right.
 *
 * Works on unsorted collections and returns the first occurrence of the target element.
 *
 * @par Characteristics:
 * - <b>Preconditions:</b> None (works on unsorted or sorted data).
 *
 * @par Time Complexity:
 * - Best Case: @c O(1) (target is at index 0)
 * - Average Case: @c O(N)
 * - Worst Case: @c O(N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct LinearSearch {
    /**
     * @brief Returns the unique identifier for this algorithm.
     *
     * @return "linearsearch"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "linearsearch";
    }

    /**
     * @brief Linearly searches for target in the span.
     *
     * @tparam T Element type supporting @c operator==.
     * @param data Mutable span of elements to search.
     * @param target Value to locate.
     * @return Index of the first matching element, or @c std::nullopt.
     */
    template <typename T>
    std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return linear_search(data, target);
    }

    /**
     * @brief Linearly searches for target in the span.
     *
     * @tparam T Element type supporting @c operator==.
     * @param data Const span of elements to search.
     * @param target Value to locate.
     * @return Index of the first matching element, or @c std::nullopt.
     */
    template <typename T>
    std::optional<std::size_t> search(std::span<const T> data, const T& target) const {
        return linear_search(data, target);
    }

    /**
     * @brief Linearly searches for target in the span using a custom binary predicate.
     *
     * @tparam T Element type.
     * @tparam BinaryPredicate Binary predicate callable testing equivalence.
     * @param data Mutable span of elements to search.
     * @param target Value to locate.
     * @param pred Binary predicate callable.
     * @return Index of the first matching element, or @c std::nullopt.
     */
    template <typename T, typename BinaryPredicate>
    std::optional<std::size_t> search(std::span<T> data, const T& target,
                                      BinaryPredicate pred) const {
        return linear_search(data, target, std::move(pred));
    }

    /**
     * @brief Linearly searches for target in the span using a custom binary predicate.
     *
     * @tparam T Element type.
     * @tparam BinaryPredicate Binary predicate callable testing equivalence.
     * @param data Const span of elements to search.
     * @param target Value to locate.
     * @param pred Binary predicate callable.
     * @return Index of the first matching element, or @c std::nullopt.
     */
    template <typename T, typename BinaryPredicate>
    std::optional<std::size_t> search(std::span<const T> data, const T& target,
                                      BinaryPredicate pred) const {
        return linear_search(data, target, std::move(pred));
    }

    /**
     * @brief Indicates whether this search algorithm requires sorted input.
     * @return @c false
     */
    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return false;
    }
};

} // namespace algoat::searching

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("searching", "linearsearch", ::algoat::searching::LinearSearch)

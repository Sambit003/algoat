/**
 * @file searching.hpp
 * @brief Umbrella header for all searching algorithms and C++20 search concepts.
 */

#pragma once

#include "algoat/searching/adaptive_binary_search.hpp"
#include "algoat/searching/binary_search.hpp"
#include "algoat/searching/eytzinger_search_tree.hpp"
#include "algoat/searching/hybrid_interpolation_search.hpp"
#include "algoat/searching/interpolation_search.hpp"
#include "algoat/searching/linear_search.hpp"

#include <concepts>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <variant>

namespace algoat::searching {

/**
 * @concept SearchAlgorithm
 * @brief Specifies the compile-time contract for searching algorithms.
 *
 * Requires:
 * 1. @c name(): String identifier convertible to @c std::string_view.
 * 2. @c search(std::span<T>, const T&): Returns index in @c std::optional<std::size_t>.
 * 3. @c requires_sorted(): Returns boolean indicating whether input data must be sorted.
 *
 * @tparam Algo Searching algorithm struct.
 * @tparam T Element type.
 */
template <typename Algo, typename T>
concept SearchAlgorithm = requires(Algo algo, std::span<T> data, const T& target) {
    { algo.name() } -> std::convertible_to<std::string_view>;
    { algo.search(data, target) } -> std::same_as<std::optional<std::size_t>>;
    { algo.requires_sorted() } -> std::same_as<bool>;
};

/**
 * @brief Variant type encapsulating all supported searching algorithm implementations.
 *
 * Used by @c algoat::core::Registry<SearchVariant> for static dispatch without virtual table
 * overhead.
 */
using SearchVariant = std::variant<LinearSearch, BinarySearch, InterpolationSearch,
                                   AdaptiveBinarySearch, HybridInterpolationSearch>;

} // namespace algoat::searching

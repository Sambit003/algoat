/**
 * @file searching.hpp
 * @brief Umbrella header for all searching algorithms and C++20 search concepts.
 */

#pragma once

#include "algoat/searching/adaptive_binary_search.hpp"
#include "algoat/searching/binary_search.hpp"
#include "algoat/searching/branchless_binary_search.hpp"
#include "algoat/searching/eytzinger_search_tree.hpp"
#include "algoat/searching/galloping_search.hpp"
#include "algoat/searching/hybrid_interpolation_search.hpp"
#include "algoat/searching/interpolation_search.hpp"
#include "algoat/searching/linear_search.hpp"
#include "algoat/searching/static_btree.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <span>
#include <string_view>
#include <variant>

namespace algoat::searching {

namespace detail {

/**
 * @brief Checks if a lower_bound iterator matches the target under comp and proj, returning its
 * index.
 */
template <typename I, typename S, typename T, typename Comp, typename Proj>
[[nodiscard]] constexpr std::optional<std::size_t>
match_lower_bound_index(I first, I it, S last, const T& target, Comp comp, Proj proj) {
    if (it != last && !std::invoke(comp, std::invoke(proj, *it), target) &&
        !std::invoke(comp, target, std::invoke(proj, *it))) {
        return static_cast<std::size_t>(std::distance(first, it));
    }
    return std::nullopt;
}

} // namespace detail

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
using SearchVariant =
    std::variant<LinearSearch, BinarySearch, InterpolationSearch, AdaptiveBinarySearch,
                 HybridInterpolationSearch, BranchlessBinarySearch, GallopingSearch>;

} // namespace algoat::searching

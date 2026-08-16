/**
 * @file searching.hpp
 * @brief Umbrella header for all searching algorithms and C++20 search concepts.
 */

#pragma once

#include <concepts>
#include <string_view>
#include <span>
#include <optional>
#include <cstddef>
#include <variant>

#include "algoat/searching/linear_search.hpp"
#include "algoat/searching/binary_search.hpp"
#include "algoat/searching/interpolation_search.hpp"

namespace algoat::searching {

/**
 * @concept SearchAlgorithm
 * @brief Specifies the compile-time contract for searching algorithms.
 * 
 * Requires:
 * 1. <tt>name()</tt>: String identifier convertible to @c std::string_view.
 * 2. <tt>search(std::span<T>, const T&)</tt>: Returns index in <tt>std::optional<std::size_t></tt>.
 * 3. <tt>requires_sorted()</tt>: Returns boolean indicating whether input data must be sorted.
 * 
 * @tparam Algo Searching algorithm struct.
 * @tparam T Element type.
 */
template<typename Algo, typename T>
concept SearchAlgorithm = requires(Algo algo, std::span<T> data, const T& target) {
    { algo.name() } -> std::convertible_to<std::string_view>;
    { algo.search(data, target) } -> std::same_as<std::optional<std::size_t>>;
    { algo.requires_sorted() } -> std::same_as<bool>;
};

/**
 * @brief Variant type encapsulating all supported searching algorithm implementations.
 * 
 * Used by <tt>algoat::core::Registry<SearchVariant></tt> for static dispatch without virtual table overhead.
 */
using SearchVariant = std::variant<LinearSearch, BinarySearch, InterpolationSearch>;

} // namespace algoat::searching

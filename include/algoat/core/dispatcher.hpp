/**
 * @file dispatcher.hpp
 * @brief Dynamic algorithm dispatcher with trait-based heuristic routing.
 *
 * Coordinates algorithm selection based on input data traits (size, sortedness ratio,
 * value types) and user-supplied configuration rules.
 */

#pragma once

#include "algoat/core/config.hpp"
#include "algoat/core/registry.hpp"
#include "algoat/core/traits.hpp"
#include "algoat/searching/searching.hpp"
#include "algoat/sorting/sorting.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace algoat::core {

template <typename Algo, typename T>
concept CanSortData = requires(Algo a, std::span<T> arr) { a.sort(arr); };

template <typename Algo, typename T>
concept CanSearchData = requires(Algo a, std::span<const T> arr, const T& t) { a.search(arr, t); };

/**
 * @class Dispatcher
 * @brief Central controller for dynamic algorithm selection and execution.
 *
 * Owns algorithm registries for sorting and searching, and implements the heuristic
 * decision tree:
 *
 *
 * @par Sorting Heuristics (@c "auto"):
 * - <b>Small Arrays</b> (<tt>N < small_threshold</tt>, default 32): @c InsertionSort
 * (<tt>O(N^2)</tt>, zero overhead).
 * - <b>Nearly Sorted</b> (sortedness ratio <tt>>= 0.90</tt> or <tt><= 0.10</tt>): @c TimSort
 * (<tt>O(N)</tt> best case on partially ordered data).
 * - <b>Large Integral Arrays</b> (<tt>N > 10,000</tt> & integral type): @c RadixSortLSD (<tt>O(N *
 * k)</tt> linear time).
 * - <b>General / Default:</b> @c IntroSort (<tt>O(N log N)</tt> hybrid
 * quicksort/heapsort/insertionsort).
 *
 *
 * @par Searching Heuristics (@c "auto"):
 * - <b>Sorted Data</b> (sortedness ratio <tt>== 1.0</tt>): @c BinarySearch (<tt>O(log N)</tt>).
 * - <b>Unsorted Data:</b> @c LinearSearch (<tt>O(N)</tt>).
 */
class Dispatcher {
    Registry<sorting::SortVariant> sort_registry_; ///< Registry of available sorting algorithms.
    Registry<searching::SearchVariant>
        search_registry_; ///< Registry of available searching algorithms.
    AlgoConfig config_;   ///< User-defined configuration preferences.

public:
    /**
     * @brief Constructs a Dispatcher with the given configuration, registering default algorithms.
     *
     * @param config Configuration options specifying algorithm preferences and fallbacks.
     */
    explicit Dispatcher(AlgoConfig config);

    /**
     * @brief Sorts a contiguous span using dynamic heuristic selection.
     *
     * Profiles @c data via <tt>analyze()</tt> in O(n) time, selects an optimal algorithm,
     * checks the registry (with fallback on missing algorithms), and executes the sort.
     *
     * @tparam T The element type in the span.
     *
     * @param data The contiguous span of elements to sort in-place.
     * @throws std::runtime_error If the selected algorithm and its fallback are unregistered.
     */
    template <typename T> void sort(std::span<T> data) const {
        DataTraits traits = analyze(data);
        std::string algo_name = config_.sorting.prefer.value_or("auto");

        if (algo_name == "auto" || algo_name.empty()) {
            if (traits.size < config_.sorting.small_threshold.value_or(32)) {
                algo_name = "insertionsort";
            } else if (traits.sortedness_ratio >= 0.9 || traits.sortedness_ratio <= 0.1) {
                algo_name = "timsort";
            } else {
                if constexpr (std::is_integral_v<T>) {
                    if (traits.size > 10000) {
                        algo_name = "radixsortlsd";
                    } else {
                        algo_name = "introsort";
                    }
                } else {
                    algo_name = "introsort";
                }
            }
        }

        if (!sort_registry_.has(algo_name)) {
            algo_name = config_.sorting.fallback.value_or("heapsort");
            if (!sort_registry_.has(algo_name)) {
                throw std::runtime_error(
                    "Requested sorting algorithm not registered and fallback missing");
            }
        }

        auto algo_variant = sort_registry_.create(algo_name);
        std::visit(
            [data](auto&& algo) {
                using AlgoType = std::remove_cvref_t<decltype(algo)>;
                if constexpr (CanSortData<AlgoType, T>) {
                    algo.sort(data);
                } else {
                    throw std::invalid_argument("Algorithm does not support this data type.");
                }
            },
            algo_variant);
    }

    /**
     * @brief Searches for a target value in a span using dynamic heuristic selection.
     *
     * Profiles @c data via <tt>analyze()</tt>, selects @c BinarySearch if data is fully sorted,
     * otherwise dispatches to @c LinearSearch (or user preferences).
     *
     * @tparam T The element type in the span.
     *
     * @param data The contiguous span of elements to search.
     *
     * @param target The value to search for.
     * @return <tt>std::optional<std::size_t></tt> Found index or @c std::nullopt.
     * @throws std::runtime_error If the selected search algorithm and its fallback are
     * unregistered.
     */
    template <typename T>
    std::optional<std::size_t> search(std::span<const T> data, const T& target) const {
        DataTraits traits = analyze(data);
        std::string algo_name = config_.searching.prefer.value_or("auto");

        if (algo_name == "auto" || algo_name.empty()) {
            if (traits.sortedness_ratio == 1.0) {
                algo_name = "binarysearch";
            } else {
                algo_name = "linearsearch";
            }
        }

        if (!search_registry_.has(algo_name)) {
            algo_name = config_.searching.fallback.value_or("linearsearch");
            if (!search_registry_.has(algo_name)) {
                throw std::runtime_error(
                    "Requested searching algorithm not registered and fallback missing");
            }
        }

        auto algo_variant = search_registry_.create(algo_name);
        return std::visit(
            [data, &target](auto&& algo) -> std::optional<std::size_t> {
                using AlgoType = std::remove_cvref_t<decltype(algo)>;
                if constexpr (CanSearchData<AlgoType, T>) {
                    return algo.search(data, target);
                } else {
                    throw std::invalid_argument("Algorithm does not support this data type.");
                }
            },
            algo_variant);
    }

    template <typename T>
    std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return search(std::span<const T>{data.data(), data.size()}, target);
    }
};

} // namespace algoat::core

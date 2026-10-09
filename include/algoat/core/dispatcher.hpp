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
#include "algoat/numerics/morton.hpp"
#include "algoat/searching/searching.hpp"
#include "algoat/sorting/boolean_sort.hpp"
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
 * - <b>Small Arrays</b> (@c N < small_threshold, default 32): @c InsertionSort
 * (@c O(N^2), zero overhead).
 * - <b>Nearly Sorted</b> (sortedness ratio @c >= 0.90 or @c <= 0.10): @c TimSort
 * (@c O(N) best case on partially ordered data).
 * - <b>Large Integral Arrays</b> (@c N > 10,000 & integral type): @c RadixSortLSD (@c O(N * k)
 * linear time).
 * - <b>General / Default:</b> @c IntroSort (@c O(N log N) hybrid
 * quicksort/heapsort/insertionsort).
 *
 *
 * @par Searching Heuristics (@c "auto"):
 * - <b>Default:</b> @c AdaptiveBinarySearch (dynamic @c O(log N) with automatic
 * invariant verification and @c O(N) fallback if monotonicity violations are detected).
 */
class Dispatcher {
    Registry<sorting::SortVariant> sort_registry_; ///< Registry of available sorting algorithms.
    Registry<searching::SearchVariant>
        search_registry_; ///< Registry of available searching algorithms.

    const AlgoConfig& config_; ///< Configuration reference injected for dispatch

public:
    /**
     * @brief Constructs a Dispatcher, registering default algorithms.
     * @param config Configuration options specifying algorithm preferences and fallbacks.
     */
    explicit Dispatcher(const AlgoConfig& config);

    /**
     * @brief Sorts a contiguous span using compile-time static dispatch or dynamic heuristics.
     *
     * Statically routes domain-specific types (e.g. @c bool via @c sort_boolean, @c std::complex
     * via @c sort_complex_morton) at compile time without runtime profiling overhead.
     * For general types, profiles @c data via @c analyze() in O(n) time, selects an optimal
     * algorithm, checks the registry (with fallback on missing algorithms), and executes the sort.
     *
     * @tparam T The element type in the span.
     *
     * @param data The contiguous span of elements to sort in-place.
     * @throws std::runtime_error If the selected algorithm and its fallback are unregistered.
     */
    template <typename T> void sort(std::span<T> data) const {
        if constexpr (IsBoolean<T>) {
            sorting::sort_boolean(data);
        } else if constexpr (IsComplex<T>) {
            numerics::sort_complex_morton(data);
        } else {
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
    }

    /**
     * @brief Searches for a target value using heuristic algorithm selection.
     *
     * Depending on whether the underlying data is sorted or not, dispatches to the most
     * appropriate searching algorithm.
     *
     * @tparam T The element type in the span.
     *
     * @param data Contiguous span of elements to search.
     * @param target The value to locate.
     * @return Index of the matching element if found.
     * @throws std::runtime_error If algorithms are unavailable.
     */
    template <typename T>
    std::optional<std::size_t> search(std::span<const T> data, const T& target) const {
        std::string algo_name = config_.searching.prefer.value_or("auto");

        if (algo_name == "auto" || algo_name.empty()) {
            algo_name = "adaptivebinarysearch";
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

    /**
     * @brief Searches for a target value using heuristic algorithm selection (mutable span).
     *
     * @tparam T The element type in the span.
     *
     * @param data Contiguous span of elements to search.
     * @param target The value to locate.
     * @return Index of the matching element if found.
     */
    template <typename T>
    std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return search(std::span<const T>{data.data(), data.size()}, target);
    }
};

} // namespace algoat::core

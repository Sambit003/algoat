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
 * - <b>Default:</b> @c AdaptiveBinarySearch (dynamic <tt>O(log N)</tt> with automatic
 * invariant verification and <tt>O(N)</tt> fallback if monotonicity violations are detected).
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

    template <typename T>
    std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return search(std::span<const T>{data.data(), data.size()}, target);
    }
};

} // namespace algoat::core

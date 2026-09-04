/**
 * @file algoat.hpp
 * @brief Main public entry point for the Algoat library.
 *
 * Provides global convenience functions for sorting and searching, as well as
 * runtime configuration management. These functions internally delegate to a
 * global @c algoat::core::Dispatcher instance.
 */

#pragma once

#include "algoat/core/config.hpp"
#include "algoat/core/dispatcher.hpp"

#include <cstddef>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>

namespace algoat {

/**
 * @brief Retrieves the thread-safe global configuration instance.
 *
 * This singleton @c AlgoConfig controls default preferences, fallback algorithms,
 * and thresholds used by @c algoat::sort and @c algoat::search.
 *
 * @return Reference to the static global @c core::AlgoConfig.
 */
inline core::AlgoConfig& get_global_config() {
    static core::AlgoConfig config;
    return config;
}

inline std::shared_mutex& get_global_config_mutex() {
    static std::shared_mutex mutex;
    return mutex;
}

inline core::Dispatcher& get_dispatcher() {
    static core::Dispatcher dispatcher(get_global_config());
    return dispatcher;
}

/**
 * @brief Loads algorithm configuration overrides from a JSON file into the global config.
 *
 *
 * @param filepath Path to the JSON configuration file.
 * @throws std::runtime_error If the file cannot be opened or contains invalid JSON.
 */
inline void load_global_config(const std::string& filepath) {
    auto config = core::load_config(filepath);

    std::unique_lock lock(get_global_config_mutex());
    get_global_config() = std::move(config);
}

/**
 * @brief Sorts a contiguous span of data in-place using dynamic algorithm dispatch.
 *
 * Analyzes the input array's characteristics (size, sortedness ratio, element type)
 * in O(n) time and dynamically dispatches to the optimal sorting algorithm according
 * to global configuration and algorithmic heuristics.
 *
 * @tparam T The element type in the span.
 *
 * @param data Contiguous span of elements to sort in-place.
 */
template <typename T> void sort(std::span<T> data) {
    std::shared_lock lock(get_global_config_mutex());
    get_dispatcher().sort(data);
}

/**
 * @brief Searches for a target value within a contiguous span using dynamic algorithm dispatch.
 *
 * Profiles the input data (e.g., whether it is fully sorted) and selects the optimal
 * searching strategy (e.g., Binary Search for sorted data, Linear Search for unsorted).
 *
 * @tparam T The element type in the span.
 *
 * @param data Contiguous span of elements to search.
 *
 * @param target The value to locate.
 * @return <tt>std::optional<std::size_t></tt> Index of the matching element if found, or @c
 * std::nullopt.
 */
template <typename T> std::optional<std::size_t> search(std::span<const T> data, const T& target) {
    std::shared_lock lock(get_global_config_mutex());
    return get_dispatcher().search(data, target);
}

template <typename T> std::optional<std::size_t> search(std::span<T> data, const T& target) {
    return search(std::span<const T>{data.data(), data.size()}, target);
}

} // namespace algoat

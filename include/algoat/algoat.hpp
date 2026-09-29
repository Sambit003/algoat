/**
 * @file algoat.hpp
 * @brief Main public entry point for the Algoat library.
 *
 * Provides global convenience functions for sorting and searching, as well as
 * runtime configuration management. These functions construct and dispatch to a
 * @c algoat::core::Dispatcher instance dynamically using the current active global config.
 */

#pragma once

#include "algoat/core/config.hpp"
#include "algoat/core/config_manager.hpp"
#include "algoat/core/dispatcher.hpp"
#include "algoat/numerics/hybrid_sfc.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace algoat {

/**
 * @brief Retrieves the thread-safe global configuration instance.
 *
 * This singleton @c AlgoConfig controls default preferences, fallback algorithms,
 * and thresholds used by @c algoat::sort and @c algoat::search.
 *
 * @return A shared_ptr to the immutable active global @c core::AlgoConfig.
 */
inline std::shared_ptr<const core::AlgoConfig> get_global_config() {
    return core::ConfigManager::instance().active_config();
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
    core::ConfigManager::instance().update_config(std::move(config));
}

inline core::Dispatcher get_dispatcher() {
    return core::Dispatcher(*get_global_config());
}

enum class SpaceFillingCurve {
    Morton,  // Maximum throughput (raw speed, bit-interleaving)
    Hilbert, // Maximum spatial locality (continuous traversal)
    Hybrid   // Balanced Pareto optimal (coarse Morton + fine Hilbert)
};

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
    get_dispatcher().sort(data);
}

/**
 * @brief Unified dispatch interface for complex spatial points.
 *
 * @tparam T The arithmetic type of the complex components.
 * @param data Contiguous span of complex elements to sort in-place.
 * @param curve The space-filling curve to use. Defaults to Morton.
 */
template <typename T>
void sort(std::span<std::complex<T>> data, SpaceFillingCurve curve = SpaceFillingCurve::Morton) {
    switch (curve) {
    case SpaceFillingCurve::Morton:
        numerics::sort_complex_morton(data);
        break;
    case SpaceFillingCurve::Hilbert:
        throw std::invalid_argument("Pure Hilbert curve sorting is not yet implemented.");
    case SpaceFillingCurve::Hybrid:
        numerics::sort_complex_hybrid(data);
        break;
    }
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
    return get_dispatcher().search(data, target);
}

template <typename T> std::optional<std::size_t> search(std::span<T> data, const T& target) {
    return search(std::span<const T>{data.data(), data.size()}, target);
}

} // namespace algoat

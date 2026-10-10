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

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
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

/**
 * @brief Constructs a Dispatcher configured with the active global settings.
 * @return Configured core::Dispatcher instance.
 */
inline core::Dispatcher get_dispatcher() {
    return core::Dispatcher(*get_global_config());
}

enum class SpaceFillingCurve {
    Morton,  // Maximum throughput (raw speed, bit-interleaving)
    Hilbert, // Maximum spatial locality (continuous traversal)
    Hybrid   // Balanced Pareto optimal (coarse Morton + fine Hilbert)
};

/**
 * @brief Sorts a C++20 random-access range in-place using dynamic algorithm dispatch.
 *
 * Supports arbitrary C++20 random-access ranges (such as @c std::vector, @c std::array,
 * subranges, and custom views), custom comparators, and projections.
 *
 * @tparam R Type satisfying @c std::ranges::random_access_range.
 * @tparam Comp Callable comparator defining a strict weak ordering.
 * @tparam Proj Callable projection to extract sorting keys from elements.
 *
 * @param range The range of elements to sort in-place.
 * @param comp Strict weak ordering comparator callable.
 * @param proj Element projection callable.
 * @return Iterator pointing to the end of the range, or @c std::ranges::dangling if passed an
 * rvalue non-borrowed container.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires(!std::is_same_v<std::remove_cvref_t<Comp>, SpaceFillingCurve> &&
             std::sortable<std::ranges::iterator_t<R>, Comp, Proj>)
constexpr std::ranges::borrowed_iterator_t<R> sort(R&& range, Comp comp = {}, Proj proj = {}) {
    return get_dispatcher().sort(std::forward<R>(range), std::move(comp), std::move(proj));
}

/**
 * @brief Sorts a contiguous span of data in-place using dynamic algorithm dispatch
 * (backward-compatibility overload).
 *
 * @tparam T The element type in the span.
 * @param data Contiguous span of elements to sort in-place.
 */
template <typename T> void sort(std::span<T> data) {
    get_dispatcher().sort(data);
}

/**
 * @brief Unified dispatch interface for complex spatial points in a random-access range.
 *
 * @tparam R Type satisfying @c std::ranges::random_access_range with complex elements.
 * @param range Range of complex elements to sort in-place.
 * @param curve The space-filling curve to use. Defaults to Morton.
 */
template <detail::contiguous_sized_range R>
    requires core::IsComplex<std::ranges::range_value_t<R>>
inline void sort(R&& range, SpaceFillingCurve curve = SpaceFillingCurve::Morton) {
    sort(detail::to_span(range), curve);
}

/**
 * @brief Unified dispatch interface for complex spatial points in a span.
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
 * @brief Searches for a target value within a C++20 random-access range using dynamic algorithm
 * dispatch.
 *
 * Profiles the input data (e.g., whether it is fully sorted) and selects the optimal
 * searching strategy (e.g., Binary Search for sorted data, Linear Search for unsorted).
 * Supports custom projections and comparators.
 *
 * @tparam R Type satisfying @c std::ranges::random_access_range.
 * @tparam T The element / target type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Callable projection.
 *
 * @param range Range of elements to search.
 * @param target The value to locate.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Index of the matching element if found, or @c std::nullopt.
 */
template <std::ranges::random_access_range R, typename T = std::ranges::range_value_t<R>,
          typename Comp = std::ranges::less, typename Proj = std::identity>
    requires std::indirect_strict_weak_order<Comp, const T*,
                                             std::projected<std::ranges::iterator_t<R>, Proj>>
std::optional<std::size_t> search(R&& range, const T& target, Comp comp = {}, Proj proj = {}) {
    return get_dispatcher().search(std::forward<R>(range), target, std::move(comp),
                                   std::move(proj));
}

/**
 * @brief Searches for a target value within a contiguous span using dynamic algorithm dispatch.
 *
 * @tparam T The element type in the span.
 * @param data Contiguous span of elements to search.
 * @param target The value to locate.
 * @return Index of the matching element if found, or @c std::nullopt.
 */
template <typename T> std::optional<std::size_t> search(std::span<const T> data, const T& target) {
    return get_dispatcher().search(data, target);
}

/**
 * @brief Searches for a target value within a mutable span using dynamic algorithm dispatch.
 *
 * @tparam T The element type in the span.
 * @param data Contiguous span of elements to search.
 * @param target The value to locate.
 * @return Index of the matching element if found, or @c std::nullopt.
 */
template <typename T> std::optional<std::size_t> search(std::span<T> data, const T& target) {
    return search(std::span<const T>{data.data(), data.size()}, target);
}

} // namespace algoat

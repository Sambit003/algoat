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
#include "algoat/geometry/convex_hull.hpp"
#include "algoat/geometry/delaunay.hpp"
#include "algoat/geometry/kdtree.hpp"
#include "algoat/geometry/lbvh.hpp"
#include "algoat/geometry/predicates.hpp"
#include "algoat/meta/policy.hpp"
#include "algoat/meta/static_sort.hpp"
#include "algoat/numerics/hybrid_sfc.hpp"
#include "algoat/searching/binary_search.hpp"
#include "algoat/searching/branchless_binary_search.hpp"
#include "algoat/searching/eytzinger_search_tree.hpp"
#include "algoat/searching/galloping_search.hpp"
#include "algoat/searching/static_btree.hpp"
#include "algoat/selection/selection.hpp"
#include "algoat/sorting/introsort.hpp"
#include "algoat/sorting/pdqsort.hpp"
#include "algoat/strings/burstsort.hpp"
#include "algoat/strings/multikey_quicksort.hpp"
#include "algoat/strings/substring_search.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>

namespace algoat {

using core::SearchHint;
using geometry::AABB;
using geometry::BVHNode;
using geometry::CircleLocation;
using geometry::convex_hull_2d_andrew;
using geometry::convex_hull_2d_chan;
using geometry::convex_hull_3d_quickhull;
using geometry::DelaunayTriangulation2D;
using geometry::incircle;
using geometry::insphere;
using geometry::KDTree;
using geometry::LBVH;
using geometry::MetricType;
using geometry::orient2d;
using geometry::orient3d;
using geometry::Orientation;
using geometry::Point2D;
using geometry::Point3D;
using geometry::Polygon2D;
using geometry::Polyhedron3D;
using geometry::Ray;
using geometry::SpatialPoint;
using geometry::Triangle;
using geometry::VoronoiCell;
using searching::branchless_lower_bound;
using searching::BranchlessBinarySearch;
using searching::EytzingerSearchTree;
using searching::galloping_search;
using searching::GallopingSearch;
using searching::StaticBTree;
using selection::nth_element;
using selection::select_optimal;
using selection::top_k_sorted;
using selection::top_k_unsorted;
using strings::find_all_substrings;
using strings::find_substring;
using strings::search_pattern;

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
    if (std::is_constant_evaluated()) {
        return sorting::pdqsort(std::forward<R>(range), std::move(comp), std::move(proj));
    }
    return get_dispatcher().sort(std::forward<R>(range), std::move(comp), std::move(proj));
}

/**
 * @brief Sorts a contiguous span of data in-place using dynamic algorithm dispatch
 * (backward-compatibility overload).
 *
 * @tparam T The element type in the span.
 * @param data Contiguous span of elements to sort in-place.
 */
template <typename T> constexpr void sort(std::span<T> data) {
    if (std::is_constant_evaluated()) {
        sorting::introsort(data);
        return;
    }
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
 * Supports custom projections, comparators, and search hints.
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
 * @param hint Search hint guiding algorithm selection.
 * @return Index of the matching element if found, or @c std::nullopt.
 */
template <std::ranges::random_access_range R, typename T = std::ranges::range_value_t<R>,
          typename Comp = std::ranges::less, typename Proj = std::identity>
    requires(!std::is_same_v<std::remove_cvref_t<Comp>, core::SearchHint> &&
             std::indirect_strict_weak_order<Comp, const T*,
                                             std::projected<std::ranges::iterator_t<R>, Proj>>)
constexpr std::optional<std::size_t> search(R&& range, const T& target, Comp comp = {},
                                            Proj proj = {},
                                            core::SearchHint hint = core::SearchHint::None) {
    if (std::is_constant_evaluated()) {
        if (hint == core::SearchHint::NearStart) {
            auto it = searching::galloping_search(std::forward<R>(range), target, comp, proj);
            return searching::detail::match_lower_bound_index(
                std::ranges::begin(range), it, std::ranges::end(range), target, comp, proj);
        }
        return searching::binary_search(std::forward<R>(range), target, std::move(comp),
                                        std::move(proj));
    }
    return get_dispatcher().search(std::forward<R>(range), target, std::move(comp), std::move(proj),
                                   hint);
}

/**
 * @brief Searches for a target value within a C++20 random-access range with a semantic hint.
 *
 * @tparam R Type satisfying @c std::ranges::random_access_range.
 * @tparam T The element / target type.
 * @param range Range of elements to search.
 * @param target Value to locate.
 * @param hint Search hint.
 * @return Index of the matching element if found, or @c std::nullopt.
 */
template <std::ranges::random_access_range R, typename T = std::ranges::range_value_t<R>>
constexpr std::optional<std::size_t> search(R&& range, const T& target, core::SearchHint hint) {
    return search(std::forward<R>(range), target, std::ranges::less{}, std::identity{}, hint);
}

/**
 * @brief Searches for a target value within a contiguous span using dynamic algorithm dispatch.
 *
 * @tparam T The element type in the span.
 * @param data Contiguous span of elements to search.
 * @param target The value to locate.
 * @param hint Search hint guiding algorithm selection.
 * @return Index of the matching element if found, or @c std::nullopt.
 */
template <typename T>
constexpr std::optional<std::size_t> search(std::span<const T> data, const T& target,
                                            core::SearchHint hint = core::SearchHint::None) {
    if (std::is_constant_evaluated()) {
        if (hint == core::SearchHint::NearStart) {
            auto it = searching::galloping_search(data.begin(), data.end(), target);
            if (it != data.end() && *it == target) {
                return static_cast<std::size_t>(it - data.begin());
            }
            return std::nullopt;
        }
        return searching::binary_search(data, target);
    }
    return get_dispatcher().search(data, target, hint);
}

/**
 * @brief Searches for a target value within a mutable span using dynamic algorithm dispatch.
 *
 * @tparam T The element type in the span.
 * @param data Contiguous span of elements to search.
 * @param target The value to locate.
 * @param hint Search hint guiding algorithm selection.
 * @return Index of the matching element if found, or @c std::nullopt.
 */
template <typename T>
constexpr std::optional<std::size_t> search(std::span<T> data, const T& target,
                                            core::SearchHint hint = core::SearchHint::None) {
    return search(std::span<const T>{data.data(), data.size()}, target, hint);
}

} // namespace algoat

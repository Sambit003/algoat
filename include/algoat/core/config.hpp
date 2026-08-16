/**
 * @file config.hpp
 * @brief Configuration structures and JSON loader for runtime algorithm behavior.
 * 
 * Defines data structures controlling algorithm selection strategies, preferred
 * algorithms, fallbacks, and small-size thresholds.
 */

#pragma once

#include <string>
#include <optional>
#include <cstddef>

namespace algoat::core {

/**
 * @struct SortingConfig
 * @brief Configuration options specific to sorting algorithms.
 */
struct SortingConfig {
    std::optional<std::string> prefer;          ///< Explicit algorithm name override (e.g., "quicksort", "timsort", or "auto").
    std::optional<std::string> fallback;        ///< Algorithm to use if the preferred algorithm is unavailable or invalid (default: "heapsort").
    std::optional<std::size_t> small_threshold; ///< Element count below which InsertionSort is preferred (default: 32).
};

/**
 * @struct SearchingConfig
 * @brief Configuration options specific to searching algorithms.
 */
struct SearchingConfig {
    std::optional<std::string> prefer;   ///< Explicit search algorithm override (e.g., "binarysearch", "linearsearch", or "auto").
    std::optional<std::string> fallback; ///< Fallback algorithm if preferred is missing (default: "linearsearch").
};

/**
 * @struct AlgoConfig
 * @brief Aggregated configuration holding sorting and searching settings.
 */
struct AlgoConfig {
    SortingConfig sorting;     ///< Sorting configuration subgroup.
    SearchingConfig searching; ///< Searching configuration subgroup.
};

/**
 * @brief Parses and loads an @c AlgoConfig from a JSON configuration file.
 * 
 * Supports schema:
 * ```json
 * {
 *   "sorting": {
 *     "prefer": "auto",
 *     "fallback": "heapsort",
 *     "small_threshold": 32
 *   },
 *   "searching": {
 *     "prefer": "auto",
 *     "fallback": "linearsearch"
 *   }
 * }
 * ```
 * 
 *
 * @param filepath Path to the JSON configuration file.
 * @return @c AlgoConfig Populated configuration structure.
 * @throws std::runtime_error If the file cannot be opened or contains invalid JSON.
 */
AlgoConfig load_config(const std::string& filepath);

} // namespace algoat::core

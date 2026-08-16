/**
 * @file config.cpp
 * @brief Implementation of JSON configuration loading for Algoat.
 */

#include "algoat/core/config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace algoat::core {

AlgoConfig load_config(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + filepath);
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }

    AlgoConfig config;

    // Parse sorting configuration options
    if (j.contains("sorting") && j["sorting"].is_object()) {
        auto& s = j["sorting"];
        if (s.contains("prefer") && s["prefer"].is_string()) {
            config.sorting.prefer = s["prefer"].get<std::string>();
        }
        if (s.contains("fallback") && s["fallback"].is_string()) {
            config.sorting.fallback = s["fallback"].get<std::string>();
        }
        if (s.contains("small_threshold") && s["small_threshold"].is_number_integer()) {
            config.sorting.small_threshold = s["small_threshold"].get<std::size_t>();
        }
    }

    // Parse searching configuration options
    if (j.contains("searching") && j["searching"].is_object()) {
        auto& s = j["searching"];
        if (s.contains("prefer") && s["prefer"].is_string()) {
            config.searching.prefer = s["prefer"].get<std::string>();
        }
        if (s.contains("fallback") && s["fallback"].is_string()) {
            config.searching.fallback = s["fallback"].get<std::string>();
        }
    }

    return config;
}

} // namespace algoat::core

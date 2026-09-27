#pragma once

#include "algoat/core/config.hpp"

#include <atomic>
#include <memory>

namespace algoat::core {

class ConfigManager {
    std::atomic<std::shared_ptr<const AlgoConfig>> active_config_;

    ConfigManager() {
        // Initialize with default config
        active_config_.store(std::make_shared<const AlgoConfig>());
    }

public:
    static ConfigManager& instance() noexcept {
        static ConfigManager manager;
        return manager;
    }

    // Wait-free read path with zero locks, zero atomic RMW
    [[nodiscard]] std::shared_ptr<const AlgoConfig> active_config() const noexcept {
        return active_config_.load(std::memory_order_acquire);
    }

    // Non-blocking write path which publishes immutable snapshot
    void update_config(AlgoConfig new_config) {
        active_config_.store(std::make_shared<const AlgoConfig>(std::move(new_config)),
                             std::memory_order_release);
    }
};

} // namespace algoat::core

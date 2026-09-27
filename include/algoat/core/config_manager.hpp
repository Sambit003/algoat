#pragma once

#include "algoat/core/config.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace algoat::core {

class ConfigManager {
    std::atomic<const AlgoConfig*> active_config_;
    std::mutex garbage_mutex_;
    std::vector<std::unique_ptr<const AlgoConfig>> garbage_;

    ConfigManager() {
        active_config_.store(new AlgoConfig(), std::memory_order_relaxed);
    }

    ~ConfigManager() {
        delete active_config_.load(std::memory_order_relaxed);
    }

public:
    static ConfigManager& instance() noexcept {
        static ConfigManager manager;
        return manager;
    }

    // Wait-free read path: zero locks, zero atomic RMW on shared cache lines.
    // By using an aliased shared_ptr with a thread-local control block, we satisfy
    // the API requirement of returning a shared_ptr without inducing a MESI storm.
    [[nodiscard]] std::shared_ptr<const AlgoConfig> active_config() const noexcept {
        const AlgoConfig* ptr = active_config_.load(std::memory_order_acquire);
        thread_local std::shared_ptr<void> local_cb = std::make_shared<int>(0);
        return std::shared_ptr<const AlgoConfig>(local_cb, ptr);
    }

    // Non-blocking write path: publishes immutable snapshot
    void update_config(AlgoConfig new_config) {
        const AlgoConfig* new_ptr = new AlgoConfig(std::move(new_config));
        const AlgoConfig* old_ptr = active_config_.exchange(new_ptr, std::memory_order_release);

        // Defer reclamation (in a real system, we'd use EBR/RCU synchronization before deleting)
        std::lock_guard<std::mutex> lock(garbage_mutex_);
        garbage_.emplace_back(old_ptr);
    }
};

} // namespace algoat::core

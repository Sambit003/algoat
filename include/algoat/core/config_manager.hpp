#pragma once

#include "algoat/core/config.hpp"

#include <atomic>
#include <bitset>
#include <cstdint>
#include <memory>
#include <vector>

namespace algoat::core {

class ConfigManager {
    static constexpr size_t max_threads = 8192;

    struct alignas(64) ThreadState {
        std::shared_ptr<void> dummy;
        ThreadState() : dummy(std::make_shared<int>(0)) {}
    };

    struct RetiredNode {
        const AlgoConfig* ptr;
        std::bitset<max_threads> active_threads_mask;
        RetiredNode* next;
    };

    std::atomic<const AlgoConfig*> active_config_;
    std::atomic<RetiredNode*> retired_list_{nullptr};
    std::atomic<int> thread_id_counter_{0};
    ThreadState thread_states_[max_threads];

    int get_thread_id() {
        thread_local int id = -1;
        if (id == -1) {
            id = thread_id_counter_.fetch_add(1, std::memory_order_relaxed);
            if (id >= (int)max_threads) {
                std::terminate(); // Thread capacity exceeded
            }
        }
        return id;
    }

    void reclaim() {
        std::bitset<max_threads> quiescent_mask;

        std::atomic_thread_fence(std::memory_order_seq_cst);

        int current_max = thread_id_counter_.load(std::memory_order_relaxed);
        for (int i = 0; i < current_max; ++i) {
            if (thread_states_[i].dummy.use_count() == 1) {
                quiescent_mask.set(i);
            }
        }

        RetiredNode* current = retired_list_.exchange(nullptr, std::memory_order_acquire);
        RetiredNode* un_reclaimable = nullptr;

        while (current) {
            RetiredNode* next = current->next;

            // Clear bits for threads that have passed through a quiescent state
            current->active_threads_mask &= ~quiescent_mask;

            if (current->active_threads_mask.none()) {
                delete current->ptr;
                delete current;
            } else {
                current->next = un_reclaimable;
                un_reclaimable = current;
            }
            current = next;
        }

        if (un_reclaimable) {
            RetiredNode* tail = un_reclaimable;
            while (tail->next)
                tail = tail->next;

            RetiredNode* old_head = retired_list_.load(std::memory_order_relaxed);
            do {
                tail->next = old_head;
            } while (!retired_list_.compare_exchange_weak(
                old_head, un_reclaimable, std::memory_order_release, std::memory_order_relaxed));
        }
    }

    ConfigManager() {
        active_config_.store(new AlgoConfig(), std::memory_order_relaxed);
    }

    ~ConfigManager() {
        delete active_config_.load(std::memory_order_relaxed);
        RetiredNode* current = retired_list_.load(std::memory_order_relaxed);
        while (current) {
            RetiredNode* next = current->next;
            delete current->ptr;
            delete current;
            current = next;
        }
    }

public:
    static ConfigManager& instance() noexcept {
        static ConfigManager manager;
        return manager;
    }

    // Wait-free read path: zero locks, zero atomic RMW on shared cache lines.
    // Uses Quiescent State Based Reclamation (QSBR) with thread-local aliased shared_ptrs.
    [[nodiscard]] std::shared_ptr<const AlgoConfig> active_config() const noexcept {
        int tid = const_cast<ConfigManager*>(this)->get_thread_id();
        ThreadState& state = const_cast<ConfigManager*>(this)->thread_states_[tid];

        // Signal entry into read-side critical section by incrementing dummy refcount
        std::shared_ptr<void> local_dummy = state.dummy;

        // Ensure signal is visible before reading global pointer
        std::atomic_thread_fence(std::memory_order_seq_cst);

        const AlgoConfig* ptr = active_config_.load(std::memory_order_acquire);

        return std::shared_ptr<const AlgoConfig>(std::move(local_dummy), ptr);
    }

    // Non-blocking write path: publishes immutable snapshot
    void update_config(AlgoConfig new_config) {
        const AlgoConfig* old_ptr = active_config_.exchange(new AlgoConfig(std::move(new_config)),
                                                            std::memory_order_release);

        RetiredNode* node = new RetiredNode{old_ptr, std::bitset<max_threads>(), nullptr};

        // Ensure new pointer is published before capturing active threads
        std::atomic_thread_fence(std::memory_order_seq_cst);

        int current_max = thread_id_counter_.load(std::memory_order_relaxed);
        for (int i = 0; i < current_max; ++i) {
            if (thread_states_[i].dummy.use_count() > 1) {
                node->active_threads_mask.set(i);
            }
        }

        RetiredNode* old_head = retired_list_.load(std::memory_order_relaxed);
        do {
            node->next = old_head;
        } while (!retired_list_.compare_exchange_weak(old_head, node, std::memory_order_release,
                                                      std::memory_order_relaxed));

        reclaim();
    }
};

} // namespace algoat::core

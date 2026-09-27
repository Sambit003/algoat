#include "algoat/core/config_manager.hpp"

#include <benchmark/benchmark.h>
#include <memory>
#include <mutex>
#include <shared_mutex>

// Legacy Shared Mutex approach
struct LegacyGlobalConfigState {
    algoat::core::AlgoConfig config;
    mutable std::shared_mutex mutex;
};

static LegacyGlobalConfigState& get_legacy_state() {
    static LegacyGlobalConfigState state;
    return state;
}

static void BM_LegacySharedMutexDispatch(benchmark::State& state) {
    auto& global_state = get_legacy_state();
    for (auto _ : state) {
        std::shared_lock<std::shared_mutex> lock(global_state.mutex);
        benchmark::DoNotOptimize(global_state.config.sorting.small_threshold);
    }
}
BENCHMARK(BM_LegacySharedMutexDispatch)->ThreadRange(1, 64)->UseRealTime();

// New Atomic Snapshot Pointer / Lock-Free RCU approach
static void BM_LockFreeRCUDispatch(benchmark::State& state) {
    auto& manager = algoat::core::ConfigManager::instance();
    for (auto _ : state) {
        auto config = manager.active_config();
        benchmark::DoNotOptimize(config->sorting.small_threshold);
    }
}
BENCHMARK(BM_LockFreeRCUDispatch)->ThreadRange(1, 64)->UseRealTime();

BENCHMARK_MAIN();

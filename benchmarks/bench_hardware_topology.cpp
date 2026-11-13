/**
 * @file bench_hardware_topology.cpp
 * @brief Microbenchmarks verifying sub-5 nanosecond query latency for cached HardwareTopology.
 */

#include <algoat/core/hardware_topology.hpp>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>

using algoat::core::HardwareTopology;

static void BM_HardwareTopology_InstanceAndL1D(benchmark::State& state) {
    for (auto _ : state) {
        std::size_t l1d = HardwareTopology::instance().l1d_cache_size();
        benchmark::DoNotOptimize(l1d);
    }
}
BENCHMARK(BM_HardwareTopology_InstanceAndL1D);

static void BM_HardwareTopology_CachedRefCacheLine(benchmark::State& state) {
    const auto& topo = HardwareTopology::instance();
    for (auto _ : state) {
        std::size_t line = topo.cache_line_size();
        benchmark::DoNotOptimize(line);
    }
}
BENCHMARK(BM_HardwareTopology_CachedRefCacheLine);

static void BM_HardwareTopology_CachedRefSimdFlags(benchmark::State& state) {
    const auto& topo = HardwareTopology::instance();
    for (auto _ : state) {
        bool flags = topo.has_avx2() || topo.has_avx512f() || topo.has_bmi2() || topo.has_neon();
        benchmark::DoNotOptimize(flags);
    }
}
BENCHMARK(BM_HardwareTopology_CachedRefSimdFlags);

static void BM_HardwareTopology_OptimalL1BlockElements(benchmark::State& state) {
    const auto& topo = HardwareTopology::instance();
    for (auto _ : state) {
        std::size_t elems = topo.optimal_l1_block_elements<std::int32_t>(0.5);
        benchmark::DoNotOptimize(elems);
    }
}
BENCHMARK(BM_HardwareTopology_OptimalL1BlockElements);

BENCHMARK_MAIN();

/**
 * @file bench_profiler.cpp
 * @brief Benchmark suite measuring latency of the sub-linear SIMD profiler and adaptive search.
 */

#include <algoat/algoat.hpp>
#include <algoat/core/traits.hpp>
#include <benchmark/benchmark.h>
#include <numeric>
#include <random>
#include <vector>

static void BM_SublinearProfiler_Sorted_1M(benchmark::State& state) {
    const std::size_t N = 1000000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    for (auto _ : state) {
        auto traits = algoat::core::analyze(std::span<const int>{data.data(), data.size()});
        benchmark::DoNotOptimize(traits);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_SublinearProfiler_Sorted_1M);

static void BM_SublinearProfiler_Unsorted_1M(benchmark::State& state) {
    const std::size_t N = 1000000;
    std::vector<int> data(N);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 10000000);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = dist(gen);
    }

    for (auto _ : state) {
        auto traits = algoat::core::analyze(std::span<const int>{data.data(), data.size()});
        benchmark::DoNotOptimize(traits);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_SublinearProfiler_Unsorted_1M);

static void BM_DispatcherSearch_Auto_Sorted_1M(benchmark::State& state) {
    const std::size_t N = 1000000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);
    const int target = 750000;

    for (auto _ : state) {
        auto res = algoat::search(std::span<const int>{data.data(), data.size()}, target);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DispatcherSearch_Auto_Sorted_1M);

static void BM_DispatcherSearch_Explicit_BinarySearch_1M(benchmark::State& state) {
    const std::size_t N = 1000000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);
    const int target = 750000;

    algoat::core::AlgoConfig config;
    config.searching.prefer = "binarysearch";
    algoat::core::Dispatcher dispatcher(config);

    for (auto _ : state) {
        auto res = dispatcher.search(std::span<const int>{data.data(), data.size()}, target);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DispatcherSearch_Explicit_BinarySearch_1M);

BENCHMARK_MAIN();

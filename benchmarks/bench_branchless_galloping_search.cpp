/**
 * @file bench_branchless_galloping_search.cpp
 * @brief Google Benchmark suite evaluating Branchless Binary Search (CMOV) and Galloping Search.
 *
 * Acceptance verification:
 * 1. Branchless search achieves >= 2.5x throughput over std::lower_bound for N <= 16,384.
 * 2. Galloping search achieves >= 3x speedup over binary search when target index k <= 100 in N =
 * 1,000,000.
 */

#include "algoat/searching/branchless_binary_search.hpp"
#include "algoat/searching/galloping_search.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

using namespace algoat::searching;

template <typename T> static std::vector<T> generate_sorted_data(std::size_t size) {
    std::vector<T> data(size);
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(i * 4);
    }
    return data;
}

template <typename T> static std::vector<T> generate_random_queries(std::size_t count, T max_val) {
    std::vector<T> queries(count);
    std::mt19937 gen(1337);
    std::uniform_int_distribution<T> dist(0, max_val);
    for (std::size_t i = 0; i < count; ++i) {
        queries[i] = dist(gen);
    }
    return queries;
}

template <typename T>
static std::vector<T> generate_near_start_queries(std::size_t count, std::size_t max_index,
                                                  const std::vector<T>& data) {
    std::vector<T> queries(count);
    std::mt19937 gen(42);
    std::uniform_int_distribution<std::size_t> dist(0, max_index);
    for (std::size_t i = 0; i < count; ++i) {
        queries[i] = data[dist(gen)];
    }
    return queries;
}

// -----------------------------------------------------------------------------
// Benchmark 1: Cache-resident branchless search vs std::lower_bound (N <= 16,384)
// -----------------------------------------------------------------------------

static void BM_StdLowerBound_CacheResident(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const auto data = generate_sorted_data<int32_t>(size);
    const auto queries = generate_random_queries<int32_t>(16384, data.back() + 10);

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto it = std::lower_bound(data.begin(), data.end(), val);
        benchmark::DoNotOptimize(it);
        q_idx = (q_idx + 1) & (queries.size() - 1);
    }
    state.SetItemsProcessed(state.iterations());
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StdLowerBound_CacheResident)->RangeMultiplier(4)->Range(64, 16384)->Complexity();

static void BM_BranchlessLowerBound_CacheResident(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const auto data = generate_sorted_data<int32_t>(size);
    const auto queries = generate_random_queries<int32_t>(16384, data.back() + 10);

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto it = branchless_lower_bound(data.begin(), data.end(), val);
        benchmark::DoNotOptimize(it);
        q_idx = (q_idx + 1) & (queries.size() - 1);
    }
    state.SetItemsProcessed(state.iterations());
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_BranchlessLowerBound_CacheResident)
    ->RangeMultiplier(4)
    ->Range(64, 16384)
    ->Complexity();

// -----------------------------------------------------------------------------
// Benchmark 2: Galloping search vs Binary search on N = 1,000,000 with k <= 100
// -----------------------------------------------------------------------------

static void BM_BinarySearch_NearStart_1M(benchmark::State& state) {
    const std::size_t max_k = static_cast<std::size_t>(state.range(0));
    const std::size_t n = 1000000;
    const auto data = generate_sorted_data<int32_t>(n);
    const auto queries = generate_near_start_queries<int32_t>(8192, max_k, data);

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto it = std::lower_bound(data.begin(), data.end(), val);
        benchmark::DoNotOptimize(it);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BinarySearch_NearStart_1M)->Arg(1)->Arg(10)->Arg(50)->Arg(100);

static void BM_GallopingSearch_NearStart_1M(benchmark::State& state) {
    const std::size_t max_k = static_cast<std::size_t>(state.range(0));
    const std::size_t n = 1000000;
    const auto data = generate_sorted_data<int32_t>(n);
    const auto queries = generate_near_start_queries<int32_t>(8192, max_k, data);

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto it = galloping_search(data.begin(), data.end(), val);
        benchmark::DoNotOptimize(it);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_GallopingSearch_NearStart_1M)->Arg(1)->Arg(10)->Arg(50)->Arg(100);

BENCHMARK_MAIN();

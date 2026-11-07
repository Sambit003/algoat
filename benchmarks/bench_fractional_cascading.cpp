/**
 * @file bench_fractional_cascading.cpp
 * @brief Benchmark comparing FractionalCascading against K independent std::lower_bound searches.
 */

#include "algoat/searching/fractional_cascading.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

using namespace algoat::searching;

static std::vector<std::vector<int64_t>> generate_lists(std::size_t k, std::size_t n,
                                                        uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, static_cast<int64_t>(n * 10));

    std::vector<std::vector<int64_t>> lists(k);
    for (std::size_t i = 0; i < k; ++i) {
        lists[i].resize(n);
        for (std::size_t j = 0; j < n; ++j) {
            lists[i][j] = dist(rng);
        }
        std::sort(lists[i].begin(), lists[i].end());
    }
    return lists;
}

static std::vector<int64_t> generate_queries(std::size_t count, int64_t max_val, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(-100, max_val + 100);
    std::vector<int64_t> queries(count);
    for (std::size_t i = 0; i < count; ++i) {
        queries[i] = dist(rng);
    }
    return queries;
}

static void BM_StdLowerBound_MultiList(benchmark::State& state) {
    const std::size_t k = static_cast<std::size_t>(state.range(0));
    const std::size_t n = static_cast<std::size_t>(state.range(1));

    const auto lists = generate_lists(k, n, 42);
    const auto queries = generate_queries(10000, static_cast<int64_t>(n * 10), 1337);

    std::vector<std::size_t> results(k);
    std::size_t q_idx = 0;

    for (auto _ : state) {
        const int64_t q = queries[q_idx];
        for (std::size_t i = 0; i < k; ++i) {
            auto it = std::lower_bound(lists[i].begin(), lists[i].end(), q);
            results[i] = static_cast<std::size_t>(it - lists[i].begin());
        }
        benchmark::DoNotOptimize(results.data());
        q_idx = (q_idx + 1) % queries.size();
    }
}

static void BM_FractionalCascading_MultiList(benchmark::State& state) {
    const std::size_t k = static_cast<std::size_t>(state.range(0));
    const std::size_t n = static_cast<std::size_t>(state.range(1));

    const auto lists = generate_lists(k, n, 42);
    const FractionalCascading<int64_t> fc(lists);
    const auto queries = generate_queries(10000, static_cast<int64_t>(n * 10), 1337);

    std::vector<std::size_t> results(k);
    std::size_t q_idx = 0;

    for (auto _ : state) {
        const int64_t q = queries[q_idx];
        fc.search_all_to_span(q, results);
        benchmark::DoNotOptimize(results.data());
        q_idx = (q_idx + 1) % queries.size();
    }
}

// Benchmark across different K values and list lengths
BENCHMARK(BM_StdLowerBound_MultiList)
    ->Args({5, 2000})
    ->Args({10, 2000})
    ->Args({20, 2000})
    ->Args({50, 2000})
    ->Args({20, 10000});

BENCHMARK(BM_FractionalCascading_MultiList)
    ->Args({5, 2000})
    ->Args({10, 2000})
    ->Args({20, 2000})
    ->Args({50, 2000})
    ->Args({20, 10000});

BENCHMARK_MAIN();

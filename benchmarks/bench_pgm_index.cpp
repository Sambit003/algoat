#include "algoat/searching/pgm_index.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

using namespace algoat::searching;

template <typename T> static std::vector<T> generate_sorted_data(std::size_t size) {
    std::vector<T> data(size);
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<T> dist(1, static_cast<T>(size * 10));
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    std::sort(data.begin(), data.end());
    return data;
}

template <typename T>
static std::vector<T> generate_random_queries(std::size_t num_queries, T max_val) {
    std::vector<T> queries(num_queries);
    std::mt19937_64 gen(1337);
    std::uniform_int_distribution<T> dist(0, max_val + 10);
    for (std::size_t i = 0; i < num_queries; ++i) {
        queries[i] = dist(gen);
    }
    return queries;
}

static void BM_StdLowerBound_Int64(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const auto data = generate_sorted_data<int64_t>(size);
    const auto queries = generate_random_queries<int64_t>(10000, data.back());

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto it = std::lower_bound(data.begin(), data.end(), val);
        benchmark::DoNotOptimize(it);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StdLowerBound_Int64)->RangeMultiplier(10)->Range(10000, 10000000)->Complexity();

static void BM_PGMLowerBound_Int64(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const auto data = generate_sorted_data<int64_t>(size);
    const PGMIndex<int64_t, 64, 16> index(data);
    const auto queries = generate_random_queries<int64_t>(10000, data.back());

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto res = index.lower_bound(data, val);
        benchmark::DoNotOptimize(res);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_PGMLowerBound_Int64)->RangeMultiplier(10)->Range(10000, 10000000)->Complexity();

BENCHMARK_MAIN();

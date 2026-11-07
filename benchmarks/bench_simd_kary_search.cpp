#include "algoat/searching/simd_kary_search.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

using namespace algoat::searching;

template <typename T> static std::vector<T> generate_sorted_data(std::size_t size) {
    std::vector<T> data(size);
    std::mt19937 gen(42);
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
    std::mt19937 gen(1337);
    std::uniform_int_distribution<T> dist(0, max_val + 10);
    for (std::size_t i = 0; i < num_queries; ++i) {
        queries[i] = dist(gen);
    }
    return queries;
}

static void BM_StdLowerBound_Int32(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const auto data = generate_sorted_data<int32_t>(size);
    const auto queries = generate_random_queries<int32_t>(10000, data.back());

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto it = std::lower_bound(data.begin(), data.end(), val);
        benchmark::DoNotOptimize(it);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}

static void BM_SimdKarySearch_Int32(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const auto data = generate_sorted_data<int32_t>(size);
    const auto queries = generate_random_queries<int32_t>(10000, data.back());
    std::span<const int32_t> s{data};

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto res = simd_kary_search(s, val);
        benchmark::DoNotOptimize(res);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}

static void BM_SIMDKarySearchTree8_Int32(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const auto data = generate_sorted_data<int32_t>(size);
    const SIMDKarySearchTree<int32_t, 8> tree(data);
    const auto queries = generate_random_queries<int32_t>(10000, data.back());

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto res = tree.lower_bound(val);
        benchmark::DoNotOptimize(res);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}

static void BM_SIMDKarySearchTree16_Int32(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    const auto data = generate_sorted_data<int32_t>(size);
    const SIMDKarySearchTree<int32_t, 16> tree(data);
    const auto queries = generate_random_queries<int32_t>(10000, data.back());

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto res = tree.lower_bound(val);
        benchmark::DoNotOptimize(res);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int n : {16, 64, 256, 1024, 4096, 16384, 65536}) {
        b->Args({n});
    }
}

BENCHMARK(BM_StdLowerBound_Int32)->Apply(CustomArguments);
BENCHMARK(BM_SimdKarySearch_Int32)->Apply(CustomArguments);
BENCHMARK(BM_SIMDKarySearchTree8_Int32)->Apply(CustomArguments);
BENCHMARK(BM_SIMDKarySearchTree16_Int32)->Apply(CustomArguments);

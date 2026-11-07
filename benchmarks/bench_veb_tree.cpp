#include "algoat/searching/veb_tree.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

using namespace algoat::searching;

static std::vector<uint32_t> generate_unique_keys(std::size_t n, uint32_t max_val, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0, max_val);
    std::set<uint32_t> unique;
    while (unique.size() < n) {
        unique.insert(dist(gen));
    }
    return std::vector<uint32_t>(unique.begin(), unique.end());
}

static std::vector<uint32_t> generate_queries(std::size_t count, uint32_t max_val, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0, max_val);
    std::vector<uint32_t> queries(count);
    for (std::size_t i = 0; i < count; ++i) {
        queries[i] = dist(gen);
    }
    return queries;
}

static void BM_StdSet_Successor(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    constexpr uint32_t max_val = 65535;
    const auto keys = generate_unique_keys(n, max_val, 42);
    const auto queries = generate_queries(10000, max_val, 1337);

    std::set<uint32_t> s(keys.begin(), keys.end());

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto it = s.upper_bound(val);
        benchmark::DoNotOptimize(it);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StdSet_Successor)->RangeMultiplier(10)->Range(100, 10000)->Complexity();

static void BM_VEBTree_Successor(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    constexpr uint32_t max_val = 65535;
    const auto keys = generate_unique_keys(n, max_val, 42);
    const auto queries = generate_queries(10000, max_val, 1337);

    VanEmdeBoasTree<16> veb;
    for (auto k : keys) {
        veb.insert(k);
    }

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto res = veb.successor(val);
        benchmark::DoNotOptimize(res);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_VEBTree_Successor)->RangeMultiplier(10)->Range(100, 10000)->Complexity();

static void BM_VEBTree_Predecessor(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    constexpr uint32_t max_val = 65535;
    const auto keys = generate_unique_keys(n, max_val, 42);
    const auto queries = generate_queries(10000, max_val, 1337);

    VanEmdeBoasTree<16> veb;
    for (auto k : keys) {
        veb.insert(k);
    }

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto val = queries[q_idx];
        auto res = veb.predecessor(val);
        benchmark::DoNotOptimize(res);
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_VEBTree_Predecessor)->RangeMultiplier(10)->Range(100, 10000)->Complexity();

BENCHMARK_MAIN();

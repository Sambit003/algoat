#include "algoat/algoat.hpp"
#include "algoat/selection/selection.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

namespace {

static std::vector<int> generate_random_data(std::size_t size) {
    std::vector<int> data(size);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 1000000);
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

} // namespace

// 1. StdSort Baseline
static void BM_StdSort(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_data(n);
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        std::sort(copy.begin(), copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StdSort)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

// 2. StdNthElement (Median)
static void BM_StdNthElement_Median(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_data(n);
    const std::size_t k = n / 2;
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        std::nth_element(copy.begin(), copy.begin() + k, copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StdNthElement_Median)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

// 3. AlgoatIntroselect_Median (Musser Introselect + BFPRT)
static void BM_AlgoatIntroselect_Median(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_data(n);
    const std::size_t k = n / 2;
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        algoat::selection::nth_element(copy.begin(), copy.begin() + k, copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_AlgoatIntroselect_Median)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

// 4. AlgoatSelectOptimal_Median (Floyd-Rivest Algorithm 489)
static void BM_AlgoatSelectOptimal_Median(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_data(n);
    const std::size_t k = n / 2;
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        algoat::selection::select_optimal(copy.begin(), copy.begin() + k, copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_AlgoatSelectOptimal_Median)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

// 5. AlgoatSelectOptimal_SmallRank (K = 10)
static void BM_AlgoatSelectOptimal_SmallRank(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_data(n);
    const std::size_t k = 10;
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        algoat::selection::select_optimal(copy.begin(), copy.begin() + k, copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_AlgoatSelectOptimal_SmallRank)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Complexity();

// 6. AlgoatTopK_Sorted (K = 50)
static void BM_AlgoatTopK_Sorted(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_data(n);
    const std::size_t k = 50;
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        algoat::selection::top_k_sorted(copy, k);
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_AlgoatTopK_Sorted)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

// 7. AlgoatTopK_Unsorted (K = 50)
static void BM_AlgoatTopK_Unsorted(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_data(n);
    const std::size_t k = 50;
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        algoat::selection::top_k_unsorted(copy, k);
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_AlgoatTopK_Unsorted)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

BENCHMARK_MAIN();

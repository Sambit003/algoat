#include "algoat/sorting/blocksort.hpp"
#include "algoat/sorting/bucketsort.hpp"
#include "algoat/sorting/combsort.hpp"
#include "algoat/sorting/countingsort.hpp"
#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/insertionsort.hpp"
#include "algoat/sorting/introsort.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "algoat/sorting/quicksort.hpp"
#include "algoat/sorting/radixsort.hpp"
#include "algoat/sorting/shellsort.hpp"
#include "algoat/sorting/timsort.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace algoat::sorting;

static std::vector<int> generate_random_data(size_t size) {
    std::vector<int> data(size);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 1000000);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

static void BM_StdSort(benchmark::State& state) {
    auto data = generate_random_data(state.range(0));
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        std::sort(copy.begin(), copy.end());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StdSort)->RangeMultiplier(2)->Range(8, 8 << 12)->Complexity();

#define DECLARE_SORT_BENCHMARK(AlgoClass)                                                          \
    static void BM_##AlgoClass(benchmark::State& state) {                                          \
        auto data = generate_random_data(state.range(0));                                          \
        AlgoClass algo;                                                                            \
        for (auto _ : state) {                                                                     \
            state.PauseTiming();                                                                   \
            auto copy = data;                                                                      \
            state.ResumeTiming();                                                                  \
            algo.sort(std::span{copy});                                                            \
        }                                                                                          \
        state.SetComplexityN(state.range(0));                                                      \
    }                                                                                              \
    BENCHMARK(BM_##AlgoClass)->RangeMultiplier(2)->Range(8, 8 << 12)->Complexity();

DECLARE_SORT_BENCHMARK(InsertionSort)
DECLARE_SORT_BENCHMARK(QuickSort)
DECLARE_SORT_BENCHMARK(MergeSort)
DECLARE_SORT_BENCHMARK(HeapSort)
DECLARE_SORT_BENCHMARK(TimSort)
DECLARE_SORT_BENCHMARK(IntroSort)
DECLARE_SORT_BENCHMARK(BlockSort)
DECLARE_SORT_BENCHMARK(RadixSortLSD)
DECLARE_SORT_BENCHMARK(RadixSortMSD)
DECLARE_SORT_BENCHMARK(RadixSortInPlaceMSD)
DECLARE_SORT_BENCHMARK(CountingSort)
DECLARE_SORT_BENCHMARK(BucketSort)
DECLARE_SORT_BENCHMARK(ShellSort)
DECLARE_SORT_BENCHMARK(CombSort)

#define DECLARE_LARGE_SORT_BENCHMARK(Name, SortCall)                                               \
    static void BM_##Name##_Large(benchmark::State& state) {                                       \
        auto data = generate_random_data(state.range(0));                                          \
        for (auto _ : state) {                                                                     \
            state.PauseTiming();                                                                   \
            auto copy = data;                                                                      \
            state.ResumeTiming();                                                                  \
            SortCall;                                                                              \
        }                                                                                          \
        state.SetComplexityN(state.range(0));                                                      \
    }                                                                                              \
    BENCHMARK(BM_##Name##_Large)->RangeMultiplier(4)->Range(1 << 16, 1 << 22)->Complexity();

DECLARE_LARGE_SORT_BENCHMARK(StdSort, std::sort(copy.begin(), copy.end()))
DECLARE_LARGE_SORT_BENCHMARK(RadixSortInPlaceMSD,
                             algoat::sorting::inplace_radix_sort(std::span{copy}))
DECLARE_LARGE_SORT_BENCHMARK(MergeSort, algoat::sorting::MergeSort{}.sort(std::span{copy}))

static void BM_MergeSort_Large_PMR(benchmark::State& state) {
    auto data = generate_random_data(state.range(0));
    std::size_t buffer_size = state.range(0) * sizeof(int) + 256;
    std::vector<std::byte> arena(buffer_size);
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        // null_memory_resource ensures no secondary dynamic allocations occur
        std::pmr::monotonic_buffer_resource mbr(arena.data(), arena.size(),
                                                std::pmr::null_memory_resource());
        state.ResumeTiming();
        algoat::sorting::mergesort(std::span{copy}, &mbr);
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_MergeSort_Large_PMR)->RangeMultiplier(4)->Range(1 << 16, 1 << 22)->Complexity();

#include <algoat/numerics/float16_sort.hpp>
#include <algoat/numerics/morton.hpp>
#include <algoat/sorting/boolean_sort.hpp>

static void BM_Bool_StdSort(benchmark::State& state) {
    std::vector<uint8_t> data(state.range(0));
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        std::sort(copy.begin(), copy.end());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Bool_StdSort)->RangeMultiplier(2)->Range(8, 8 << 12)->Complexity();

static void BM_Bool_AlgoatSort(benchmark::State& state) {
    std::vector<uint8_t> data(state.range(0));
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        algoat::sorting::sort_boolean(std::span<uint8_t>{copy});
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Bool_AlgoatSort)->RangeMultiplier(2)->Range(8, 8 << 12)->Complexity();

// Semantic alias for 16-bit float layout
using Float16 = uint16_t;

static void BM_Float16_StdSort(benchmark::State& state) {
    std::vector<Float16> data(state.range(0));
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint16_t> dist(0, 65535);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        std::sort(copy.begin(), copy.end(), algoat::numerics::Float16Compare{});
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Float16_StdSort)->RangeMultiplier(2)->Range(8, 8 << 12)->Complexity();

static void BM_Float16_AlgoatSort(benchmark::State& state) {
    std::vector<Float16> data(state.range(0));
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint16_t> dist(0, 65535);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        algoat::numerics::sort_float16(std::span<Float16>{copy});
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Float16_AlgoatSort)->RangeMultiplier(2)->Range(8, 8 << 12)->Complexity();

static void BM_Complex_StdSort(benchmark::State& state) {
    std::vector<std::complex<float>> data(state.range(0));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = {dist(gen), dist(gen)};
    }
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        std::sort(copy.begin(), copy.end(), algoat::numerics::MortonCompare{});
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complex_StdSort)->RangeMultiplier(2)->Range(8, 8 << 12)->Complexity();

static void BM_Complex_AlgoatSort(benchmark::State& state) {
    std::vector<std::complex<float>> data(state.range(0));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = {dist(gen), dist(gen)};
    }
    for (auto _ : state) {
        state.PauseTiming();
        auto copy = data;
        state.ResumeTiming();
        algoat::numerics::sort_complex_morton(std::span<std::complex<float>>{copy});
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complex_AlgoatSort)->RangeMultiplier(2)->Range(8, 8 << 12)->Complexity();

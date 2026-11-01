#include "algoat/sketches/count_min.hpp"
#include "algoat/sketches/detail/hash.hpp"

#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

using namespace algoat::sketches;

static void BM_CountMinSketch_Add_Int_CU(benchmark::State& state) {
    CountMinSketch<uint64_t, uint32_t> cms(0.001, 0.01, true);

    uint64_t key = 0;
    for (auto _ : state) {
        cms.add(key++);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CountMinSketch_Add_Int_CU);

static void BM_CountMinSketch_Add_Int_Standard(benchmark::State& state) {
    CountMinSketch<uint64_t, uint32_t> cms(0.001, 0.01, false);

    uint64_t key = 0;
    for (auto _ : state) {
        cms.add(key++);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CountMinSketch_Add_Int_Standard);

static void BM_CountMinSketch_Estimate(benchmark::State& state) {
    CountMinSketch<uint64_t, uint32_t> cms(0.001, 0.01, true);
    for (uint64_t i = 0; i < 50000; ++i) {
        cms.add(i, 1);
    }

    uint64_t key = 0;
    for (auto _ : state) {
        auto val = cms.estimate(key++);
        benchmark::DoNotOptimize(val);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CountMinSketch_Estimate);

static void BM_CountMinSketch_Merge(benchmark::State& state) {
    CountMinSketch<uint64_t, uint32_t> a(0.001, 0.01, false);
    CountMinSketch<uint64_t, uint32_t> b(0.001, 0.01, false);

    for (uint64_t i = 0; i < 10000; ++i) {
        a.add(i, 2);
        b.add(i + 5000, 3);
    }

    for (auto _ : state) {
        CountMinSketch<uint64_t, uint32_t> tmp = a;
        tmp.merge(b);
        benchmark::DoNotOptimize(tmp);
    }
}
BENCHMARK(BM_CountMinSketch_Merge);

static void BM_SpaceSaving_Add_Int(benchmark::State& state) {
    SpaceSavingHeavyHitters<uint64_t, 100> ss;

    uint64_t key = 0;
    for (auto _ : state) {
        key = (key * 2654435761ULL) ^ 0x9e3779b97f4a7c15ULL;
        ss.add(key % 500, 1);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SpaceSaving_Add_Int);

static void BM_SpaceSaving_TopK(benchmark::State& state) {
    SpaceSavingHeavyHitters<uint64_t, 100> ss;
    for (uint64_t i = 0; i < 10000; ++i) {
        ss.add(i % 200, 1);
    }

    for (auto _ : state) {
        auto items = ss.top_k();
        benchmark::DoNotOptimize(items);
    }
}
BENCHMARK(BM_SpaceSaving_TopK);

#include "algoat/sketches/bloom_cuckoo_filter.hpp"

#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

using namespace algoat::sketches;

static void BM_BlockedBloomFilter_Insert(benchmark::State& state) {
    constexpr size_t N = 1000000;
    BlockedBloomFilter<uint64_t> filter(N, 0.01);

    uint64_t key = 0;
    for (auto _ : state) {
        filter.insert(key++);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BlockedBloomFilter_Insert);

static void BM_BlockedBloomFilter_Contains(benchmark::State& state) {
    constexpr size_t N = 1000000;
    BlockedBloomFilter<uint64_t> filter(N, 0.01);
    for (size_t i = 0; i < N; ++i) {
        filter.insert(i * 2);
    }

    uint64_t key = 0;
    for (auto _ : state) {
        bool res = filter.contains(key++);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BlockedBloomFilter_Contains);

static void BM_BlockedBloomFilter_BatchContains(benchmark::State& state) {
    constexpr size_t N = 1000000;
    BlockedBloomFilter<uint64_t> filter(N, 0.01);
    for (size_t i = 0; i < N; ++i) {
        filter.insert(i * 2);
    }

    constexpr size_t batch_size = 64;
    std::vector<uint64_t> keys(batch_size);
    std::array<bool, batch_size> results{};
    uint64_t counter = 0;

    for (auto _ : state) {
        for (size_t i = 0; i < batch_size; ++i) {
            keys[i] = counter++;
        }
        filter.contains_batch(keys, results);
        benchmark::DoNotOptimize(results);
    }
    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_BlockedBloomFilter_BatchContains);

static void BM_CuckooFilter_Insert(benchmark::State& state) {
    constexpr size_t N = 100000;
    CuckooFilter<uint64_t, 16, 4> filter(N * 2);

    uint64_t key = 0;
    for (auto _ : state) {
        bool ok = filter.insert(key++);
        benchmark::DoNotOptimize(ok);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CuckooFilter_Insert);

static void BM_CuckooFilter_Contains(benchmark::State& state) {
    constexpr size_t N = 100000;
    CuckooFilter<uint64_t, 16, 4> filter(N * 2);
    for (size_t i = 0; i < N; ++i) {
        (void)filter.insert(i * 2);
    }

    uint64_t key = 0;
    for (auto _ : state) {
        bool res = filter.contains(key++);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CuckooFilter_Contains);

static void BM_CuckooFilter_Remove(benchmark::State& state) {
    constexpr size_t N = 100000;
    CuckooFilter<uint64_t, 16, 4> filter(N * 2);
    for (size_t i = 0; i < N; ++i) {
        (void)filter.insert(i);
    }

    uint64_t key = 0;
    for (auto _ : state) {
        bool res = filter.remove(key++);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CuckooFilter_Remove);

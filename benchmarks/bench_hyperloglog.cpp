#include "algoat/sketches/detail/hash.hpp"
#include "algoat/sketches/hyperloglog.hpp"

#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

using namespace algoat::sketches;

namespace {

template <std::size_t P = 14>
HyperLogLog<P> make_prepopulated_hll(std::size_t items, uint64_t seed = 0) {
    HyperLogLog<P> hll;
    for (std::size_t i = 0; i < items; ++i) {
        hll.add(seed + i);
    }
    return hll;
}

} // namespace

static void BM_HyperLogLog_Add_Int(benchmark::State& state) {
    auto hll = make_prepopulated_hll<14>(20000);

    uint64_t counter = 0;
    for (auto _ : state) {
        hll.add(counter++);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HyperLogLog_Add_Int);

static void BM_HyperLogLog_Add_Hash(benchmark::State& state) {
    auto hll = make_prepopulated_hll<14>(20000);

    uint64_t hash = 0x123456789abcdef0ULL;
    for (auto _ : state) {
        hash = detail::splitmix64(hash);
        hll.add_hash(hash);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HyperLogLog_Add_Hash);

static void BM_HyperLogLog_Estimate_P14(benchmark::State& state) {
    auto hll = make_prepopulated_hll<14>(50000);

    for (auto _ : state) {
        double est = hll.estimate();
        benchmark::DoNotOptimize(est);
    }
}
BENCHMARK(BM_HyperLogLog_Estimate_P14);

static void BM_HyperLogLog_Merge_InPlace_Dense_P14(benchmark::State& state) {
    auto a = make_prepopulated_hll<14>(20000, 0);
    auto b = make_prepopulated_hll<14>(20000, 10000);

    for (auto _ : state) {
        a.merge(b);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_HyperLogLog_Merge_InPlace_Dense_P14);

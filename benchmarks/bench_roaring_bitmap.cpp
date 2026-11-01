#include "algoat/succinct/roaring_bitmap.hpp"

#include <benchmark/benchmark.h>
#include <vector>

using namespace algoat::succinct;

static void BM_SimdOps_Bitset_AND_Kernel(benchmark::State& state) {
    alignas(32) std::array<uint64_t, 1024> a{};
    alignas(32) std::array<uint64_t, 1024> b{};
    alignas(32) std::array<uint64_t, 1024> out{};
    a.fill(0xAAAAAAAAAAAAAAAAULL);
    b.fill(0x5555555555555555ULL);

    for (auto _ : state) {
        detail::SimdOps::bitset_and(out.data(), a.data(), b.data());
        benchmark::DoNotOptimize(out.data());
    }

    // Effective bandwidth: 2 read streams (8192 + 8192) + 1 write stream (8192) = 24576 bytes
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * 24576);
}
BENCHMARK(BM_SimdOps_Bitset_AND_Kernel);

static void BM_Bitset_Intersection_AND(benchmark::State& state) {
    RoaringBitmap a;
    RoaringBitmap b;
    // Populate chunk 0 dense bitset containers (each 65536 bits = 8192 bytes)
    for (uint32_t i = 0; i < 65536; i += 2) {
        a.add(i);
    }
    for (uint32_t i = 0; i < 65536; i += 3) {
        b.add(i);
    }

    for (auto _ : state) {
        RoaringBitmap res = a & b;
        benchmark::DoNotOptimize(res);
    }

    // Set throughput: 2 input bitsets of 8192 bytes = 16384 bytes per cycle
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * 16384);
}
BENCHMARK(BM_Bitset_Intersection_AND);

static void BM_Cardinality_1M(benchmark::State& state) {
    RoaringBitmap bm;
    // Populate 1,000,000 integers across multiple containers
    for (uint32_t i = 0; i < 1000000; ++i) {
        bm.add(i * 3);
    }

    for (auto _ : state) {
        uint64_t card = bm.cardinality();
        benchmark::DoNotOptimize(card);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 1000000);
}
BENCHMARK(BM_Cardinality_1M);

static void BM_Jaccard_Index_AllocationFree(benchmark::State& state) {
    RoaringBitmap a;
    RoaringBitmap b;
    for (uint32_t i = 0; i < 100000; ++i) {
        a.add(i * 2);
        b.add(i * 3);
    }

    for (auto _ : state) {
        double j = RoaringBitmap::jaccard_index(a, b);
        benchmark::DoNotOptimize(j);
    }
}
BENCHMARK(BM_Jaccard_Index_AllocationFree);

static void BM_Serialize_Deserialize(benchmark::State& state) {
    RoaringBitmap bm;
    for (uint32_t i = 0; i < 50000; ++i) {
        bm.add(i * 2);
    }

    std::vector<uint8_t> bytes = bm.serialize();

    for (auto _ : state) {
        RoaringBitmap restored = RoaringBitmap::deserialize(bytes);
        benchmark::DoNotOptimize(restored);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(bytes.size()));
}
BENCHMARK(BM_Serialize_Deserialize);

#include "algoat/succinct/rank_select.hpp"

#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace algoat::succinct;

namespace {

RankSelectBitVector create_benchmark_bitvector(size_t n, double density) {
    std::mt19937_64 rng(42);
    size_t words_count = (n + 63) / 64;
    std::vector<uint64_t> words(words_count, 0ULL);
    uint64_t threshold = static_cast<uint64_t>(density * static_cast<double>(~0ULL));

    for (size_t i = 0; i < words_count; ++i) {
        // Generate pseudo-random word with target density
        uint64_t w = 0;
        for (size_t b = 0; b < 64; ++b) {
            if (rng() <= threshold) {
                w |= (1ULL << b);
            }
        }
        words[i] = w;
    }
    return RankSelectBitVector(std::span<const uint64_t>{words}, n);
}

} // namespace

static void BM_Rank1_Sequential_L1Hot(benchmark::State& state) {
    constexpr size_t N = 1000000; // 1M bits fits in L2/L3 cache
    auto bv = create_benchmark_bitvector(N, 0.5);

    size_t pos = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(bv.rank1(pos));
        pos += 64;
        if (pos >= N)
            pos = 0;
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Rank1_Sequential_L1Hot);

static void BM_Rank1_Random(benchmark::State& state) {
    constexpr size_t N = 10000000; // 10M bits
    auto bv = create_benchmark_bitvector(N, 0.5);

    std::mt19937_64 rng(123);
    std::vector<size_t> query_positions(65536);
    for (size_t& p : query_positions) {
        p = rng() % N;
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(bv.rank1(query_positions[q_idx]));
        q_idx = (q_idx + 1) & 65535;
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Rank1_Random);

static void BM_Select1_Sequential_L1Hot(benchmark::State& state) {
    constexpr size_t N = 1000000;
    auto bv = create_benchmark_bitvector(N, 0.5);
    uint64_t total_ones = bv.total_ones();

    uint64_t k = 1;
    for (auto _ : state) {
        benchmark::DoNotOptimize(bv.select1(k));
        k += 64;
        if (k > total_ones)
            k = 1;
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Select1_Sequential_L1Hot);

static void BM_Select1_Queries(benchmark::State& state) {
    constexpr size_t N = 1000000;
    auto bv = create_benchmark_bitvector(N, 0.5);
    uint64_t total_ones = bv.total_ones();

    std::mt19937_64 rng(456);
    std::vector<uint64_t> query_k(65536);
    for (uint64_t& k : query_k) {
        k = 1 + (rng() % total_ones);
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(bv.select1(query_k[q_idx]));
        q_idx = (q_idx + 1) & 65535;
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Select1_Queries);

static void BM_Select0_Queries(benchmark::State& state) {
    constexpr size_t N = 1000000;
    auto bv = create_benchmark_bitvector(N, 0.5);
    uint64_t total_zeros = N - bv.total_ones();

    std::mt19937_64 rng(789);
    std::vector<uint64_t> query_k(65536);
    for (uint64_t& k : query_k) {
        k = 1 + (rng() % total_zeros);
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(bv.select0(query_k[q_idx]));
        q_idx = (q_idx + 1) & 65535;
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Select0_Queries);

static void BM_Broadword_SelectInWord(benchmark::State& state) {
    uint64_t word = 0xAAAAAAAAAAAAAAAAULL;
    uint32_t k = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(detail::select64(word, k));
        k = (k + 1) & 31;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Broadword_SelectInWord);

#include "algoat/succinct/fm_index.hpp"

#include <benchmark/benchmark.h>
#include <random>
#include <string>
#include <vector>

namespace {

using algoat::succinct::FMIndex;
using algoat::succinct::FMIndexConfig;

std::string generate_random_dna(size_t n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    const char bases[] = {'A', 'C', 'G', 'T'};
    std::uniform_int_distribution<size_t> dist(0, 3);
    std::string s;
    s.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(bases[dist(rng)]);
    }
    return s;
}

} // namespace

static void BM_FMIndex_Construction(benchmark::State& state) {
    const size_t n = state.range(0);
    const std::string text = generate_random_dna(n);

    for (auto _ : state) {
        FMIndex idx{text};
        benchmark::DoNotOptimize(idx);
    }
    state.SetBytesProcessed(state.iterations() * n * sizeof(char));
}
BENCHMARK(BM_FMIndex_Construction)->RangeMultiplier(4)->Range(1024, 1048576);

static void BM_FMIndex_Count_DNA(benchmark::State& state) {
    constexpr size_t N = 100'000;
    const std::string text = generate_random_dna(N);
    const FMIndex idx{text};

    const size_t pat_len = state.range(0);
    std::mt19937_64 rng(12345);
    std::vector<std::string> patterns(1024);
    for (auto& pat : patterns) {
        size_t pos = rng() % (N - pat_len);
        pat = text.substr(pos, pat_len);
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(idx.count(patterns[q_idx]));
        q_idx = (q_idx + 1) & 1023;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FMIndex_Count_DNA)->Arg(10)->Arg(20)->Arg(50);

static void BM_FMIndex_Count_DNA_LargeScale_10M(benchmark::State& state) {
    // Acceptance criterion: count(P) for |P|=20 on DNA text |T|=10^7 in < 1 us, >1M queries/sec
    constexpr size_t N = 10'000'000;
    const std::string text = generate_random_dna(N);
    const FMIndex idx{text};

    constexpr size_t pat_len = 20;
    std::mt19937_64 rng(98765);
    std::vector<std::string> patterns(1024);
    for (auto& pat : patterns) {
        size_t pos = rng() % (N - pat_len);
        pat = text.substr(pos, pat_len);
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(idx.count(patterns[q_idx]));
        q_idx = (q_idx + 1) & 1023;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FMIndex_Count_DNA_LargeScale_10M)->Iterations(100000);

static void BM_FMIndex_Locate(benchmark::State& state) {
    constexpr size_t N = 100'000;
    const std::string text = generate_random_dna(N);
    FMIndexConfig cfg;
    cfg.sa_sample_rate = state.range(0);
    const FMIndex idx{text, cfg};

    std::mt19937_64 rng(54321);
    constexpr size_t pat_len = 12;
    std::vector<std::string> patterns(512);
    for (auto& pat : patterns) {
        size_t pos = rng() % (N - pat_len);
        pat = text.substr(pos, pat_len);
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        auto locs = idx.locate(patterns[q_idx]);
        benchmark::DoNotOptimize(locs.data());
        q_idx = (q_idx + 1) & 511;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FMIndex_Locate)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

static void BM_FMIndex_Extract(benchmark::State& state) {
    constexpr size_t N = 100'000;
    const std::string text = generate_random_dna(N);
    const FMIndex idx{text};

    const size_t len = state.range(0);
    std::mt19937_64 rng(9999);
    std::vector<size_t> query_positions(1024);
    for (size_t& pos : query_positions) {
        pos = rng() % (N - len);
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        std::string s = idx.extract(query_positions[q_idx], len);
        benchmark::DoNotOptimize(s.data());
        q_idx = (q_idx + 1) & 1023;
    }
    state.SetBytesProcessed(state.iterations() * len);
}
BENCHMARK(BM_FMIndex_Extract)->Arg(10)->Arg(50)->Arg(200);

#include "algoat/succinct/elias_fano.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace algoat::succinct;

namespace {

EliasFanoList create_benchmark_list(size_t n, uint64_t universe) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, universe - 1);

    std::vector<uint64_t> seq(n);
    for (size_t i = 0; i < n; ++i) {
        seq[i] = dist(rng);
    }
    std::sort(seq.begin(), seq.end());

    EliasFanoEncoder encoder(n, universe);
    for (uint64_t x : seq) {
        encoder.push(x);
    }
    return EliasFanoList(encoder.finalize());
}

} // namespace

static void BM_EliasFano_At_Sequential_L1Hot(benchmark::State& state) {
    constexpr size_t N = 100000;
    constexpr uint64_t U = 10000000ULL;
    auto list = create_benchmark_list(N, U);

    size_t i = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(list.at(i));
        ++i;
        if (i >= N)
            i = 0;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_EliasFano_At_Sequential_L1Hot);

static void BM_EliasFano_NextGeq_Random(benchmark::State& state) {
    constexpr size_t N = 100000;
    constexpr uint64_t U = 10000000ULL;
    auto list = create_benchmark_list(N, U);

    std::mt19937_64 rng(9999);
    std::vector<uint64_t> query_vals(65536);
    for (uint64_t& v : query_vals) {
        v = rng() % U;
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(list.next_geq(query_vals[q_idx]));
        q_idx = (q_idx + 1) & 65535;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_EliasFano_NextGeq_Random);

static void BM_EliasFano_Iteration_Throughput(benchmark::State& state) {
    constexpr size_t N = 100000;
    constexpr uint64_t U = 10000000ULL;
    auto list = create_benchmark_list(N, U);

    for (auto _ : state) {
        uint64_t sum = 0;
        for (uint64_t val : list) {
            sum += val;
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_EliasFano_Iteration_Throughput);

#include "algoat/succinct/wavelet_matrix.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace algoat::succinct;

namespace {

std::vector<uint8_t> create_random_data_u8(size_t n, uint8_t max_val = 255) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint16_t> dist(0, max_val);
    std::vector<uint8_t> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<uint8_t>(dist(rng));
    }
    return data;
}

std::vector<uint32_t> create_random_data_u32(size_t n, uint32_t max_val = 10000) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, max_val);
    std::vector<uint32_t> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

} // namespace

static void BM_WaveletMatrix_Construction_U8(benchmark::State& state) {
    const size_t n = state.range(0);
    auto data = create_random_data_u8(n);

    for (auto _ : state) {
        const WaveletMatrix<uint8_t> wm(data);
        benchmark::DoNotOptimize(wm);
    }
    state.SetBytesProcessed(state.iterations() * n * sizeof(uint8_t));
}
BENCHMARK(BM_WaveletMatrix_Construction_U8)->Range(1024, 1048576);

static void BM_WaveletMatrix_Access_Random_U8(benchmark::State& state) {
    constexpr size_t N = 100000;
    auto data = create_random_data_u8(N);
    const WaveletMatrix<uint8_t> wm(data);

    std::mt19937_64 rng(1234);
    std::vector<size_t> query_indices(65536);
    for (size_t& idx : query_indices) {
        idx = rng() % N;
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(wm.access(query_indices[q_idx]));
        q_idx = (q_idx + 1) & 65535;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WaveletMatrix_Access_Random_U8);

static void BM_WaveletMatrix_Rank_Random_U8(benchmark::State& state) {
    constexpr size_t N = 100000;
    auto data = create_random_data_u8(N);
    const WaveletMatrix<uint8_t> wm(data);

    std::mt19937_64 rng(5678);
    struct Query {
        uint8_t c;
        size_t pos;
    };
    std::vector<Query> queries(65536);
    for (auto& q : queries) {
        q.c = static_cast<uint8_t>(rng() % 256);
        q.pos = rng() % (N + 1);
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(wm.rank(queries[q_idx].c, queries[q_idx].pos));
        q_idx = (q_idx + 1) & 65535;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WaveletMatrix_Rank_Random_U8);

static void BM_WaveletMatrix_Select_Random_U8(benchmark::State& state) {
    constexpr size_t N = 100000;
    auto data = create_random_data_u8(N);
    const WaveletMatrix<uint8_t> wm(data);

    std::mt19937_64 rng(9999);
    struct Query {
        uint8_t c;
        size_t k;
    };
    std::vector<Query> queries;
    queries.reserve(65536);
    while (queries.size() < 65536) {
        uint8_t c = static_cast<uint8_t>(rng() % 256);
        size_t total = wm.rank(c, N);
        if (total > 0) {
            queries.push_back({c, (rng() % total) + 1});
        }
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(wm.select(queries[q_idx].c, queries[q_idx].k));
        q_idx = (q_idx + 1) & 65535;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WaveletMatrix_Select_Random_U8);

static void BM_WaveletMatrix_RangeQuantile_U32(benchmark::State& state) {
    constexpr size_t N = 100000;
    auto data = create_random_data_u32(N);
    const WaveletMatrix<uint32_t> wm(data);

    std::mt19937_64 rng(8888);
    struct Query {
        size_t l;
        size_t r;
        size_t k;
    };
    std::vector<Query> queries(65536);
    for (auto& q : queries) {
        size_t a = rng() % N;
        size_t b = rng() % N;
        if (a > b)
            std::swap(a, b);
        if (a == b) {
            if (b < N)
                ++b;
            else if (a > 0)
                --a;
        }
        q.l = a;
        q.r = b;
        q.k = (b > a) ? (rng() % (b - a)) : 0;
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            wm.range_quantile(queries[q_idx].l, queries[q_idx].r, queries[q_idx].k));
        q_idx = (q_idx + 1) & 65535;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WaveletMatrix_RangeQuantile_U32);

static void BM_WaveletMatrix_RangeCount_U32(benchmark::State& state) {
    constexpr size_t N = 100000;
    auto data = create_random_data_u32(N);
    const WaveletMatrix<uint32_t> wm(data);

    std::mt19937_64 rng(7777);
    struct Query {
        size_t l;
        size_t r;
        uint32_t lo;
        uint32_t hi;
    };
    std::vector<Query> queries(65536);
    for (auto& q : queries) {
        size_t a = rng() % N;
        size_t b = rng() % N;
        if (a > b)
            std::swap(a, b);
        q.l = a;
        q.r = b;
        uint32_t v1 = rng() % 10000;
        uint32_t v2 = rng() % 10000;
        q.lo = std::min(v1, v2);
        q.hi = std::max(v1, v2);
    }

    size_t q_idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(wm.range_count(queries[q_idx].l, queries[q_idx].r,
                                                queries[q_idx].lo, queries[q_idx].hi));
        q_idx = (q_idx + 1) & 65535;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WaveletMatrix_RangeCount_U32);

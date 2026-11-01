#include "algoat/sketches/lsh.hpp"

#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

using namespace algoat::sketches;

// ============================================================================
// 1. Acceptance Criterion 2: BBitMinHash Signature Comparison (< 4 ns per pair)
// ============================================================================

static void BM_BBitMinHash_SignatureComparison_K256_B2(benchmark::State& state) {
    using MinHash = BBitMinHash<256, 2>;
    MinHash generator(42);

    std::vector<uint64_t> doc1 = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<uint64_t> doc2 = {10, 20, 30, 40, 55, 65, 75, 85, 95, 105};

    auto sig1 = generator.compute_signature(doc1);
    auto sig2 = generator.compute_signature(doc2);

    for (auto _ : state) {
        double sim = sig1.jaccard_similarity(sig2);
        benchmark::DoNotOptimize(sim);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BBitMinHash_SignatureComparison_K256_B2);

static void BM_BBitMinHash_HammingDistance_K256_B2(benchmark::State& state) {
    using MinHash = BBitMinHash<256, 2>;
    MinHash generator(42);

    std::vector<uint64_t> doc1 = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    std::vector<uint64_t> doc2 = {10, 20, 30, 40, 55, 65, 75, 85, 95, 105};

    auto sig1 = generator.compute_signature(doc1);
    auto sig2 = generator.compute_signature(doc2);

    for (auto _ : state) {
        size_t dist = sig1.hamming_distance(sig2);
        benchmark::DoNotOptimize(dist);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BBitMinHash_HammingDistance_K256_B2);

static void BM_BBitMinHash_ComputeSignature(benchmark::State& state) {
    using MinHash = BBitMinHash<256, 2>;
    MinHash generator(12345);

    const size_t num_tokens = static_cast<size_t>(state.range(0));
    std::vector<uint64_t> tokens(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        tokens[i] = i * 0x9E3779B97F4A7C15ULL + 1;
    }

    for (auto _ : state) {
        auto sig = generator.compute_signature(tokens);
        benchmark::DoNotOptimize(sig);
    }
    state.SetItemsProcessed(state.iterations() * num_tokens);
}
BENCHMARK(BM_BBitMinHash_ComputeSignature)->RangeMultiplier(4)->Range(64, 4096);

// ============================================================================
// 2. Acceptance Criterion 2: SimHash Generation Throughput (> 20M tokens/sec)
// ============================================================================

static void BM_SimHash_Compute64_Throughput(benchmark::State& state) {
    const size_t num_tokens = static_cast<size_t>(state.range(0));
    std::vector<uint64_t> tokens(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        tokens[i] = (i + 1) * 0xBF58476D1CE4E5B9ULL;
    }

    for (auto _ : state) {
        uint64_t fp = SimHash::compute_64(tokens);
        benchmark::DoNotOptimize(fp);
    }
    state.SetItemsProcessed(state.iterations() * num_tokens);
}
BENCHMARK(BM_SimHash_Compute64_Throughput)->RangeMultiplier(4)->Range(256, 16384);

static void BM_SimHash_CosineSimilarity(benchmark::State& state) {
    uint64_t fp1 = 0xAAAAAAAAAAAAAAAAULL;
    uint64_t fp2 = 0x5555555555555555ULL;

    for (auto _ : state) {
        double sim = SimHash::cosine_similarity(fp1, fp2);
        benchmark::DoNotOptimize(sim);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SimHash_CosineSimilarity);

// ============================================================================
// 3. Banded LSHIndex Insertion and Querying
// ============================================================================

static void BM_LSHIndex_Insert(benchmark::State& state) {
    using MinHash = BBitMinHash<256, 2>;
    MinHash generator(999);
    LSHIndex<MinHash::Signature, 16, 16> index;

    std::vector<uint64_t> doc = {1, 2, 3, 4, 5, 6, 7, 8};
    auto sig = generator.compute_signature(doc);

    uint64_t id = 0;
    for (auto _ : state) {
        index.insert(id++, sig);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LSHIndex_Insert);

static void BM_LSHIndex_QueryCandidates(benchmark::State& state) {
    using MinHash = BBitMinHash<256, 2>;
    MinHash generator(999);
    LSHIndex<MinHash::Signature, 16, 16> index;

    constexpr size_t N = 10000;
    for (size_t i = 0; i < N; ++i) {
        std::vector<uint64_t> doc = {i, i + 1, i + 2, i + 3};
        auto sig = generator.compute_signature(doc);
        index.insert(static_cast<uint64_t>(i), sig);
    }

    std::vector<uint64_t> query_doc = {500, 501, 502, 503};
    auto query_sig = generator.compute_signature(query_doc);

    std::vector<uint64_t> candidates;
    for (auto _ : state) {
        candidates.clear();
        index.query_candidates(query_sig, candidates);
        benchmark::DoNotOptimize(candidates);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LSHIndex_QueryCandidates);

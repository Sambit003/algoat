#include <algoat/dp/alignment.hpp>
#include <benchmark/benchmark.h>
#include <random>
#include <string>
#include <vector>

namespace {

using algoat::dp::GapPenalty;
using algoat::dp::needleman_wunsch;
using algoat::dp::ScoringMatrix;
using algoat::dp::smith_waterman_batch;
using algoat::dp::smith_waterman_score;

std::string generate_protein(size_t len, uint32_t seed = 42) {
    constexpr std::string_view alphabet = "ACDEFGHIKLMNPQRSTVWY";
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, alphabet.size() - 1);
    std::string s;
    s.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        s.push_back(alphabet[dist(rng)]);
    }
    return s;
}

static void BM_Scalar_SmithWaterman_1000x1000(benchmark::State& state) {
    const auto q = generate_protein(1000, 101);
    const auto r = generate_protein(1000, 202);
    const auto matrix = ScoringMatrix::blosum62();

    for (auto _ : state) {
        int64_t score = algoat::dp::detail::smith_waterman_scalar_score(q, r, matrix, 5, 1);
        benchmark::DoNotOptimize(score);
    }
    state.SetBytesProcessed(state.iterations() * 1000 * 1000);
}
BENCHMARK(BM_Scalar_SmithWaterman_1000x1000);

static void BM_FarrarSIMD_SmithWaterman_1000x1000(benchmark::State& state) {
    const auto q = generate_protein(1000, 101);
    const auto r = generate_protein(1000, 202);
    const auto matrix = ScoringMatrix::blosum62();
    const GapPenalty gap{-5, -1};

    for (auto _ : state) {
        int64_t score = smith_waterman_score(q, r, matrix, gap);
        benchmark::DoNotOptimize(score);
    }
    state.SetBytesProcessed(state.iterations() * 1000 * 1000);
}
BENCHMARK(BM_FarrarSIMD_SmithWaterman_1000x1000);

static void BM_Hirschberg_NeedlemanWunsch_1000x1000(benchmark::State& state) {
    const auto q = generate_protein(1000, 101);
    const auto r = generate_protein(1000, 202);
    const auto matrix = ScoringMatrix::blosum62();
    const GapPenalty gap{-5, -1};

    for (auto _ : state) {
        auto res = needleman_wunsch(q, r, matrix, gap);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * 1000 * 1000);
}
BENCHMARK(BM_Hirschberg_NeedlemanWunsch_1000x1000);

static void BM_FarrarSIMD_Batch_1000x50Targets(benchmark::State& state) {
    const auto q = generate_protein(1000, 101);
    std::vector<std::string> targets;
    targets.reserve(50);
    for (size_t i = 0; i < 50; ++i) {
        targets.push_back(generate_protein(1000, 300 + static_cast<uint32_t>(i)));
    }
    std::vector<std::string_view> target_views;
    target_views.reserve(50);
    for (const auto& t : targets) {
        target_views.emplace_back(t);
    }

    const auto matrix = ScoringMatrix::blosum62();
    const GapPenalty gap{-5, -1};

    for (auto _ : state) {
        auto scores = smith_waterman_batch(q, target_views, matrix, gap);
        benchmark::DoNotOptimize(scores);
    }
    state.SetBytesProcessed(state.iterations() * 50 * 1000 * 1000);
}
BENCHMARK(BM_FarrarSIMD_Batch_1000x50Targets);

} // namespace

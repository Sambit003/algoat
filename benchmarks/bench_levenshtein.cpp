#include "algoat/dp/levenshtein.hpp"

#include <benchmark/benchmark.h>
#include <random>
#include <string>
#include <vector>

namespace {

// Naive O(MN) Wagner-Fischer DP for speedup comparison
size_t naive_levenshtein(std::string_view s1, std::string_view s2) {
    const size_t m = s1.size();
    const size_t n = s2.size();
    std::vector<size_t> prev(m + 1);
    std::vector<size_t> curr(m + 1);
    for (size_t i = 0; i <= m; ++i)
        prev[i] = i;

    for (size_t j = 1; j <= n; ++j) {
        curr[0] = j;
        for (size_t i = 1; i <= m; ++i) {
            const size_t cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            curr[i] = std::min({curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost});
        }
        std::swap(prev, curr);
    }
    return prev[m];
}

std::string generate_random_string(size_t len, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist('a', 'z');
    std::string s;
    s.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        s.push_back(static_cast<char>(dist(rng)));
    }
    return s;
}

static void BM_NaiveDP_Len64_Text10k(benchmark::State& state) {
    std::string pattern = generate_random_string(64, 101);
    std::string text = generate_random_string(10000, 202);

    for (auto _ : state) {
        size_t d = naive_levenshtein(pattern, text);
        benchmark::DoNotOptimize(d);
    }
    state.SetBytesProcessed(state.iterations() * 64 * 10000);
}
BENCHMARK(BM_NaiveDP_Len64_Text10k);

static void BM_Myers_Len64_Text10k(benchmark::State& state) {
    std::string pattern = generate_random_string(64, 101);
    std::string text = generate_random_string(10000, 202);

    for (auto _ : state) {
        size_t d = algoat::dp::levenshtein_distance(pattern, text);
        benchmark::DoNotOptimize(d);
    }
    state.SetBytesProcessed(state.iterations() * 64 * 10000);
}
BENCHMARK(BM_Myers_Len64_Text10k);

static void BM_Myers_Len256_Text10k(benchmark::State& state) {
    std::string pattern = generate_random_string(256, 303);
    std::string text = generate_random_string(10000, 404);

    for (auto _ : state) {
        size_t d = algoat::dp::levenshtein_distance(pattern, text);
        benchmark::DoNotOptimize(d);
    }
    state.SetBytesProcessed(state.iterations() * 256 * 10000);
}
BENCHMARK(BM_Myers_Len256_Text10k);

static void BM_Ukkonen_WithinK_CloseMatch(benchmark::State& state) {
    std::string pattern = generate_random_string(1000, 505);
    std::string text = pattern;
    text[50] = 'z';
    text[500] = 'y';

    for (auto _ : state) {
        size_t d = algoat::dp::levenshtein_within_k(pattern, text, 5);
        benchmark::DoNotOptimize(d);
    }
}
BENCHMARK(BM_Ukkonen_WithinK_CloseMatch);

static void BM_Ukkonen_WithinK_EarlyExit(benchmark::State& state) {
    std::string pattern = generate_random_string(1000, 606);
    std::string text = generate_random_string(1000, 707); // completely different

    for (auto _ : state) {
        size_t d = algoat::dp::levenshtein_within_k(pattern, text, 5);
        benchmark::DoNotOptimize(d);
    }
}
BENCHMARK(BM_Ukkonen_WithinK_EarlyExit);

static void BM_AVX2_Batch_4Texts(benchmark::State& state) {
    std::string pattern = generate_random_string(64, 808);
    std::string t0 = generate_random_string(10000, 809);
    std::string t1 = generate_random_string(10000, 810);
    std::string t2 = generate_random_string(10000, 811);
    std::string t3 = generate_random_string(10000, 812);
    std::vector<std::string_view> texts = {t0, t1, t2, t3};

    for (auto _ : state) {
        auto d = algoat::dp::levenshtein_batch(pattern, texts);
        benchmark::DoNotOptimize(d);
    }
    state.SetBytesProcessed(state.iterations() * 4 * 64 * 10000);
}
BENCHMARK(BM_AVX2_Batch_4Texts);

static void BM_LCS_Length_1000(benchmark::State& state) {
    std::string s1 = generate_random_string(1000, 901);
    std::string s2 = generate_random_string(1000, 902);

    for (auto _ : state) {
        size_t l = algoat::dp::lcs_length(s1, s2);
        benchmark::DoNotOptimize(l);
    }
}
BENCHMARK(BM_LCS_Length_1000);

static void BM_Alignment_Traceback_Len64(benchmark::State& state) {
    std::string pattern = generate_random_string(64, 111);
    std::string text = generate_random_string(64, 222);

    for (auto _ : state) {
        auto a = algoat::dp::levenshtein_align(pattern, text);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_Alignment_Traceback_Len64);

} // namespace

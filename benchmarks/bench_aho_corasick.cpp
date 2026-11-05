#include "algoat/dp/aho_corasick.hpp"

#include <benchmark/benchmark.h>
#include <random>
#include <string>
#include <vector>

namespace {

std::string generate_random_ascii_text(size_t len, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    // Printable ASCII text: space (32) to tilde (126)
    std::uniform_int_distribution<int> dist(32, 126);
    std::string s;
    s.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        s.push_back(static_cast<char>(dist(rng)));
    }
    return s;
}

std::vector<std::string> generate_random_patterns(size_t k, size_t max_len, uint32_t seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> char_dist('a', 'z');
    std::uniform_int_distribution<size_t> len_dist(4, max_len);

    std::vector<std::string> patterns;
    patterns.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        size_t len = len_dist(rng);
        std::string p;
        p.reserve(len);
        for (size_t j = 0; j < len; ++j) {
            p.push_back(static_cast<char>(char_dist(rng)));
        }
        patterns.push_back(std::move(p));
    }
    return patterns;
}

// ----------------------------------------------------------------------------
// Acceptance Criteria 3:
// Text scanning at > 1 GB/s single-threaded for K = 1,000 patterns of length <= 10 on ASCII text.
// ----------------------------------------------------------------------------
static void BM_AhoCorasick_Scan_1k_Patterns_AsciiText(benchmark::State& state) {
    constexpr size_t kNumPatterns = 1000;
    constexpr size_t kMaxLen = 10;
    constexpr size_t kTextLen = 1024 * 1024; // 1 MB text

    auto patterns = generate_random_patterns(kNumPatterns, kMaxLen, 555);
    auto text = generate_random_ascii_text(kTextLen, 777);

    algoat::dp::AhoCorasick<char> ac;
    for (const auto& p : patterns) {
        ac.add_keyword(p);
    }
    ac.build();

    for (auto _ : state) {
        auto matches = ac.find_all(text);
        benchmark::DoNotOptimize(matches);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kTextLen);
}
BENCHMARK(BM_AhoCorasick_Scan_1k_Patterns_AsciiText);

static void BM_AhoCorasick_Callback_1k_Patterns_AsciiText(benchmark::State& state) {
    constexpr size_t kNumPatterns = 1000;
    constexpr size_t kMaxLen = 10;
    constexpr size_t kTextLen = 1024 * 1024;

    auto patterns = generate_random_patterns(kNumPatterns, kMaxLen, 555);
    auto text = generate_random_ascii_text(kTextLen, 777);

    algoat::dp::AhoCorasick<char> ac;
    for (const auto& p : patterns) {
        ac.add_keyword(p);
    }
    ac.build();

    for (auto _ : state) {
        size_t count = 0;
        ac.find_all_callback(text, [&](algoat::dp::AcMatch m) {
            benchmark::DoNotOptimize(m);
            ++count;
            return true;
        });
        benchmark::DoNotOptimize(count);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kTextLen);
}
BENCHMARK(BM_AhoCorasick_Callback_1k_Patterns_AsciiText);

static void BM_AhoCorasick_FindFirst_1k_Patterns(benchmark::State& state) {
    constexpr size_t kNumPatterns = 1000;
    constexpr size_t kMaxLen = 10;
    constexpr size_t kTextLen = 1024 * 1024;

    auto patterns = generate_random_patterns(kNumPatterns, kMaxLen, 555);
    auto text = generate_random_ascii_text(kTextLen, 777);

    algoat::dp::AhoCorasick<char> ac;
    for (const auto& p : patterns) {
        ac.add_keyword(p);
    }
    ac.build();

    for (auto _ : state) {
        auto match = ac.find_first(text);
        benchmark::DoNotOptimize(match);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kTextLen);
}
BENCHMARK(BM_AhoCorasick_FindFirst_1k_Patterns);

static void BM_AhoCorasick_Scan_SparseAsciiText_SimdBypass(benchmark::State& state) {
    constexpr size_t kNumPatterns = 1000;
    constexpr size_t kMaxLen = 10;
    constexpr size_t kTextLen = 1024 * 1024; // 1 MB text

    // Patterns starting with 4 prefix characters (e.g., specific protocol commands or tokens)
    std::mt19937 rng(555);
    std::uniform_int_distribution<int> char_dist('a', 'z');
    std::uniform_int_distribution<size_t> len_dist(4, kMaxLen);
    std::uniform_int_distribution<int> prefix_dist(0, 3);
    const char prefixes[] = {'X', 'Y', 'Z', 'W'};

    std::vector<std::string> patterns;
    patterns.reserve(kNumPatterns);
    for (size_t i = 0; i < kNumPatterns; ++i) {
        size_t len = len_dist(rng);
        std::string p;
        p.push_back(prefixes[prefix_dist(rng)]);
        for (size_t j = 1; j < len; ++j) {
            p.push_back(static_cast<char>(char_dist(rng)));
        }
        patterns.push_back(std::move(p));
    }

    auto text = generate_random_ascii_text(kTextLen, 777);

    algoat::dp::AhoCorasick<char> ac;
    for (const auto& p : patterns) {
        ac.add_keyword(p);
    }
    ac.build();

    for (auto _ : state) {
        auto matches = ac.find_all(text);
        benchmark::DoNotOptimize(matches);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kTextLen);
}
BENCHMARK(BM_AhoCorasick_Scan_SparseAsciiText_SimdBypass);

} // namespace

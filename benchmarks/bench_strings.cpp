#include "algoat/strings/burstsort.hpp"
#include "algoat/strings/multikey_quicksort.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Generator for synthetic Wikipedia titles with shared prefixes and namespaces
std::vector<std::string> generate_wikipedia_titles(size_t count) {
    std::vector<std::string> titles;
    titles.reserve(count);
    std::mt19937_64 rng(1337);

    const std::vector<std::string> namespaces = {
        "Wikipedia:", "Talk:", "User:", "Category:", "Portal:", "File:", "Template:", "Help:"};

    const std::vector<std::string> prefix_words = {
        "List_of_",    "History_of_",  "Geography_of_", "Economy_of_", "Demographics_of_",
        "Culture_of_", "Politics_of_", "Index_of_",     "Outline_of_", "Timeline_of_"};

    const std::vector<std::string> topic_words = {
        "Computer_science",   "Quantum_mechanics", "Artificial_intelligence",
        "Machine_learning",   "United_States",     "United_Kingdom",
        "Ancient_Rome",       "French_Revolution", "Solar_System",
        "General_relativity", "Molecular_biology", "Algorithms"};

    for (size_t i = 0; i < count; ++i) {
        std::string s;
        if (rng() % 3 == 0) {
            s += namespaces[rng() % namespaces.size()];
        }
        if (rng() % 2 == 0) {
            s += prefix_words[rng() % prefix_words.size()];
        }
        s += topic_words[rng() % topic_words.size()];
        s += "_" + std::to_string(rng() % 10000);
        titles.push_back(std::move(s));
    }
    return titles;
}

// Generator for Genomic DNA sequences
std::vector<std::string> generate_genomic_dna(size_t count, size_t length = 40) {
    std::vector<std::string> dna;
    dna.reserve(count);
    std::mt19937_64 rng(999);
    const char bases[] = {'A', 'C', 'G', 'T'};

    for (size_t i = 0; i < count; ++i) {
        std::string s;
        s.reserve(length);
        for (size_t j = 0; j < length; ++j) {
            s.push_back(bases[rng() % 4]);
        }
        dna.push_back(std::move(s));
    }
    return dna;
}

// Generator for URL datasets
std::vector<std::string> generate_urls(size_t count) {
    std::vector<std::string> urls;
    urls.reserve(count);
    std::mt19937_64 rng(42);

    const std::vector<std::string> domains = {"https://www.google.com/search?q=",
                                              "https://en.wikipedia.org/wiki/",
                                              "https://github.com/Sambit003/algoat/issues/",
                                              "https://arxiv.org/abs/",
                                              "https://docs.python.org/3/library/",
                                              "https://en.cppreference.com/w/cpp/algorithm/"};

    for (size_t i = 0; i < count; ++i) {
        std::string u = domains[rng() % domains.size()];
        u += "item_" + std::to_string(rng() % 50000);
        urls.push_back(std::move(u));
    }
    return urls;
}

// 1. Wikipedia Titles Benchmark
static void BM_Wikipedia_StdSort(benchmark::State& state) {
    const size_t count = state.range(0);
    auto titles = generate_wikipedia_titles(count);
    std::vector<std::string_view> views(titles.begin(), titles.end());

    for (auto _ : state) {
        state.PauseTiming();
        auto copy = views;
        state.ResumeTiming();
        std::sort(copy.begin(), copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(count);
}
BENCHMARK(BM_Wikipedia_StdSort)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

static void BM_Wikipedia_MultiKeyQuicksort(benchmark::State& state) {
    const size_t count = state.range(0);
    auto titles = generate_wikipedia_titles(count);
    std::vector<std::string_view> views(titles.begin(), titles.end());

    for (auto _ : state) {
        state.PauseTiming();
        auto copy = views;
        state.ResumeTiming();
        algoat::strings::multikey_quicksort(std::span<std::string_view>{copy});
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(count);
}
BENCHMARK(BM_Wikipedia_MultiKeyQuicksort)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

static void BM_Wikipedia_Burstsort(benchmark::State& state) {
    const size_t count = state.range(0);
    auto titles = generate_wikipedia_titles(count);
    std::vector<std::string_view> views(titles.begin(), titles.end());

    for (auto _ : state) {
        state.PauseTiming();
        auto copy = views;
        state.ResumeTiming();
        algoat::strings::burstsort(std::span<std::string_view>{copy});
        benchmark::DoNotOptimize(copy.data());
    }
    state.SetComplexityN(count);
}
BENCHMARK(BM_Wikipedia_Burstsort)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

// 2. Genomic DNA Benchmark
static void BM_DNA_StdSort(benchmark::State& state) {
    const size_t count = state.range(0);
    auto dna = generate_genomic_dna(count);
    std::vector<std::string_view> views(dna.begin(), dna.end());

    for (auto _ : state) {
        state.PauseTiming();
        auto copy = views;
        state.ResumeTiming();
        std::sort(copy.begin(), copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
}
BENCHMARK(BM_DNA_StdSort)->Range(10000, 100000);

static void BM_DNA_MultiKeyQuicksort(benchmark::State& state) {
    const size_t count = state.range(0);
    auto dna = generate_genomic_dna(count);
    std::vector<std::string_view> views(dna.begin(), dna.end());

    for (auto _ : state) {
        state.PauseTiming();
        auto copy = views;
        state.ResumeTiming();
        algoat::strings::multikey_quicksort(std::span<std::string_view>{copy});
        benchmark::DoNotOptimize(copy.data());
    }
}
BENCHMARK(BM_DNA_MultiKeyQuicksort)->Range(10000, 100000);

static void BM_DNA_Burstsort(benchmark::State& state) {
    const size_t count = state.range(0);
    auto dna = generate_genomic_dna(count);
    std::vector<std::string_view> views(dna.begin(), dna.end());

    for (auto _ : state) {
        state.PauseTiming();
        auto copy = views;
        state.ResumeTiming();
        algoat::strings::burstsort(std::span<std::string_view>{copy});
        benchmark::DoNotOptimize(copy.data());
    }
}
BENCHMARK(BM_DNA_Burstsort)->Range(10000, 100000);

// 3. URLs Benchmark
static void BM_URLs_StdSort(benchmark::State& state) {
    const size_t count = state.range(0);
    auto urls = generate_urls(count);
    std::vector<std::string_view> views(urls.begin(), urls.end());

    for (auto _ : state) {
        state.PauseTiming();
        auto copy = views;
        state.ResumeTiming();
        std::sort(copy.begin(), copy.end());
        benchmark::DoNotOptimize(copy.data());
    }
}
BENCHMARK(BM_URLs_StdSort)->Range(10000, 100000);

static void BM_URLs_MultiKeyQuicksort(benchmark::State& state) {
    const size_t count = state.range(0);
    auto urls = generate_urls(count);
    std::vector<std::string_view> views(urls.begin(), urls.end());

    for (auto _ : state) {
        state.PauseTiming();
        auto copy = views;
        state.ResumeTiming();
        algoat::strings::multikey_quicksort(std::span<std::string_view>{copy});
        benchmark::DoNotOptimize(copy.data());
    }
}
BENCHMARK(BM_URLs_MultiKeyQuicksort)->Range(10000, 100000);

static void BM_URLs_Burstsort(benchmark::State& state) {
    const size_t count = state.range(0);
    auto urls = generate_urls(count);
    std::vector<std::string_view> views(urls.begin(), urls.end());

    for (auto _ : state) {
        state.PauseTiming();
        auto copy = views;
        state.ResumeTiming();
        algoat::strings::burstsort(std::span<std::string_view>{copy});
        benchmark::DoNotOptimize(copy.data());
    }
}
BENCHMARK(BM_URLs_Burstsort)->Range(10000, 100000);

} // namespace

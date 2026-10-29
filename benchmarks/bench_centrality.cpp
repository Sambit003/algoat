/**
 * @file bench_centrality.cpp
 * @brief Google Benchmark suite for PageRank and Brandes Betweenness Centrality engines.
 */

#include "algoat/graph/centrality.hpp"
#include "algoat/graph/csr_graph.hpp"

#include <benchmark/benchmark.h>
#include <random>
#include <utility>
#include <vector>

using namespace algoat::graph;

static std::vector<std::pair<VertexId, VertexId>>
generate_random_edges(std::size_t num_v, std::size_t num_e, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<VertexId> dist(0, static_cast<VertexId>(num_v - 1));

    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(num_e);
    for (std::size_t i = 0; i < num_e; ++i) {
        edges.emplace_back(dist(rng), dist(rng));
    }
    return edges;
}

// ---------------------------------------------------------------------------
// 1. PageRank Benchmark
// ---------------------------------------------------------------------------

static void BM_PageRank(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_random_edges(num_v, num_e);
    const auto graph = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/true);

    PageRankConfig config;
    config.damping_factor = 0.85;
    config.tolerance = 1e-4;
    config.max_iterations = 20;

    for (auto _ : state) {
        auto result = pagerank(graph, config);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * num_e);
}

BENCHMARK(BM_PageRank)
    ->Args({10000, 16})
    ->Args({100000, 16})
    ->Args({1000000, 20})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// 2. Brandes Betweenness Centrality Benchmark
// ---------------------------------------------------------------------------

static void BM_BetweennessCentrality_Unweighted(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_random_edges(num_v, num_e);
    const auto graph = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/true);

    for (auto _ : state) {
        auto result = betweenness_centrality(graph);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * (num_v * num_e));
}

BENCHMARK(BM_BetweennessCentrality_Unweighted)
    ->Args({250, 8})
    ->Args({500, 8})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

/**
 * @file bench_bipartite_matching.cpp
 * @brief Benchmark suite for Hopcroft-Karp Maximum Bipartite Matching vs Dinic Max-Flow reduction.
 *
 * Verifies Issue #90 performance targets:
 * 1. Solves matching on bipartite graph with |L| = |R| = 100,000, |E| = 1,000,000 in < 45 ms.
 * 2. Speedup over generic Dinic max-flow reduction: >= 4x faster.
 */

#include "algoat/graph/bipartite_matching.hpp"
#include "algoat/graph/max_flow.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

using namespace algoat::graph;

namespace {

// Helper to generate a random bipartite graph with fixed edge count
struct GraphData {
    std::size_t nl;
    std::size_t nr;
    std::vector<std::pair<VertexId, VertexId>> edges;
};

GraphData generate_bipartite_graph(std::size_t nl, std::size_t nr, std::size_t target_edges,
                                   uint64_t seed = 42) {
    GraphData data;
    data.nl = nl;
    data.nr = nr;
    data.edges.reserve(target_edges);

    std::mt19937_64 rng(seed);

    // Ensure degree at least 1 for all vertices if possible
    for (VertexId u = 0; u < nl && data.edges.size() < target_edges; ++u) {
        VertexId v = static_cast<VertexId>(rng() % nr);
        data.edges.emplace_back(u, v);
    }
    for (VertexId v = 0; v < nr && data.edges.size() < target_edges; ++v) {
        VertexId u = static_cast<VertexId>(rng() % nl);
        data.edges.emplace_back(u, v);
    }

    while (data.edges.size() < target_edges) {
        VertexId u = static_cast<VertexId>(rng() % nl);
        VertexId v = static_cast<VertexId>(rng() % nr);
        data.edges.emplace_back(u, v);
    }

    return data;
}

// Global cached large instance: |L| = |R| = 100,000, |E| = 1,000,000
const GraphData& get_large_instance() {
    static const GraphData instance = generate_bipartite_graph(100000, 100000, 1000000, 12345);
    return instance;
}

} // namespace

// ============================================================================
// 1. Hopcroft-Karp on Target Instance (|L|=|R|=100,000, |E|=1,000,000)
// ============================================================================

static void BM_HopcroftKarp_Target100k(benchmark::State& state) {
    const auto& data = get_large_instance();

    BipartiteGraph graph(data.nl, data.nr);
    for (const auto& [u, v] : data.edges) {
        graph.add_edge(u, v);
    }

    for (auto _ : state) {
        auto result = graph.maximum_matching();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * data.edges.size());
}
BENCHMARK(BM_HopcroftKarp_Target100k)->Unit(benchmark::kMillisecond);

// ============================================================================
// 2. Generic Dinic Flow Reduction Comparison on Same Target Instance
// ============================================================================

static void BM_DinicFlowReduction_Target100k(benchmark::State& state) {
    const auto& data = get_large_instance();

    const VertexId s = static_cast<VertexId>(data.nl + data.nr);
    const VertexId t = static_cast<VertexId>(data.nl + data.nr + 1);

    MaxFlowEngine<int64_t> flow_engine(data.nl + data.nr + 2);
    for (VertexId u = 0; u < data.nl; ++u) {
        flow_engine.add_edge(s, u, 1);
    }
    for (VertexId v = 0; v < data.nr; ++v) {
        flow_engine.add_edge(static_cast<VertexId>(data.nl + v), t, 1);
    }
    for (const auto& [u, v] : data.edges) {
        flow_engine.add_edge(u, static_cast<VertexId>(data.nl + v), 1);
    }

    for (auto _ : state) {
        auto result = flow_engine.dinic(s, t);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * data.edges.size());
}
BENCHMARK(BM_DinicFlowReduction_Target100k)->Unit(benchmark::kMillisecond);

// ============================================================================
// 3. Hopcroft-Karp Scaling Across Partition Sizes (Average Degree = 10)
// ============================================================================

static void BM_HopcroftKarp_Scaling(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t num_edges = n * 10;
    const auto data = generate_bipartite_graph(n, n, num_edges, 42);

    BipartiteGraph graph(data.nl, data.nr);
    for (const auto& [u, v] : data.edges) {
        graph.add_edge(u, v);
    }

    for (auto _ : state) {
        auto result = graph.maximum_matching();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * num_edges);
}
BENCHMARK(BM_HopcroftKarp_Scaling)
    ->RangeMultiplier(10)
    ->Range(1000, 100000)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// 4. König Minimum Vertex Cover & Maximum Independent Set Latency
// ============================================================================

static void BM_Konig_VertexCoverAndIndependentSet(benchmark::State& state) {
    const auto& data = get_large_instance();

    BipartiteGraph graph(data.nl, data.nr);
    for (const auto& [u, v] : data.edges) {
        graph.add_edge(u, v);
    }
    const auto matching = graph.maximum_matching();

    for (auto _ : state) {
        auto cover = graph.minimum_vertex_cover(matching);
        auto indep = graph.maximum_independent_set(matching);
        benchmark::DoNotOptimize(cover);
        benchmark::DoNotOptimize(indep);
    }
}
BENCHMARK(BM_Konig_VertexCoverAndIndependentSet)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

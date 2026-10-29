/**
 * @file bench_mst.cpp
 * @brief Google Benchmark suite for Parallel Boruvka-Kruskal vs Serial Kruskal MST engines.
 */

#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/mst.hpp"

#include <benchmark/benchmark.h>
#include <random>
#include <tuple>
#include <vector>

using namespace algoat::graph;

static std::vector<std::tuple<VertexId, VertexId, double>>
generate_random_sparse_graph(std::size_t num_v, std::size_t avg_deg, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<VertexId> dist(0, static_cast<VertexId>(num_v - 1));
    std::uniform_real_distribution<double> weight_dist(1.0, 1000.0);

    const std::size_t num_e = num_v * avg_deg;
    std::vector<std::tuple<VertexId, VertexId, double>> edges;
    edges.reserve(num_e);

    // Initial spanning backbone
    for (VertexId i = 1; i < num_v; ++i) {
        std::uniform_int_distribution<VertexId> prev_dist(0, i - 1);
        edges.emplace_back(prev_dist(rng), i, weight_dist(rng));
    }

    // Additional cross edges
    for (std::size_t i = num_v - 1; i < num_e; ++i) {
        VertexId u = dist(rng);
        VertexId v = dist(rng);
        if (u != v) {
            edges.emplace_back(u, v, weight_dist(rng));
        }
    }
    return edges;
}

static void BM_MST_Serial_Kruskal(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));

    const auto edges = generate_random_sparse_graph(num_v, avg_deg);
    const auto graph =
        CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(graph);

    for (auto _ : state) {
        auto result = engine.compute_serial();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * (num_v * avg_deg));
}

static void BM_MST_Parallel_BoruvkaKruskal(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto threads = static_cast<std::size_t>(state.range(2));

    const auto edges = generate_random_sparse_graph(num_v, avg_deg);
    const auto graph =
        CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(graph);

    for (auto _ : state) {
        auto result = engine.compute_parallel(threads);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * (num_v * avg_deg));
}

BENCHMARK(BM_MST_Serial_Kruskal)
    ->Args({10000, 8})
    ->Args({100000, 8})
    ->Args({1000000, 8})
    ->Args({10000000, 5})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MST_Parallel_BoruvkaKruskal)
    ->Args({10000, 8, 1})
    ->Args({10000, 8, 4})
    ->Args({10000, 8, 8})
    ->Args({100000, 8, 1})
    ->Args({100000, 8, 4})
    ->Args({100000, 8, 8})
    ->Args({1000000, 8, 8})
    ->Args({10000000, 5, 8})
    ->Unit(benchmark::kMillisecond);

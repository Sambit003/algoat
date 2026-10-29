/**
 * @file bench_shortest_path.cpp
 * @brief Google Benchmark suite for 4-Ary Heap Dijkstra vs std::priority_queue and ALT Search.
 */

#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/d_ary_heap.hpp"
#include "algoat/graph/shortest_path.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

using namespace algoat::graph;

namespace {

// Generates a synthetic road network with planar-like degree (avg degree ~ 3.5 to 4.0)
std::vector<std::tuple<VertexId, VertexId, double>> generate_road_network(std::size_t grid_dim,
                                                                          uint64_t seed = 42) {
    const std::size_t num_v = grid_dim * grid_dim;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_w(1.0, 20.0);

    auto node_id = [grid_dim](std::size_t r, std::size_t c) {
        return static_cast<VertexId>(r * grid_dim + c);
    };

    std::vector<std::tuple<VertexId, VertexId, double>> edges;
    edges.reserve(num_v * 4);

    for (std::size_t r = 0; r < grid_dim; ++r) {
        for (std::size_t c = 0; c < grid_dim; ++c) {
            VertexId u = node_id(r, c);
            if (r + 1 < grid_dim) {
                edges.emplace_back(u, node_id(r + 1, c), dist_w(rng));
            }
            if (c + 1 < grid_dim) {
                edges.emplace_back(u, node_id(r, c + 1), dist_w(rng));
            }
            // Add diagonal shortcuts with 20% probability to mimic real road intersections
            if (r + 1 < grid_dim && c + 1 < grid_dim && (rng() % 5 == 0)) {
                edges.emplace_back(u, node_id(r + 1, c + 1), dist_w(rng) * 1.414);
            }
        }
    }
    return edges;
}

// Naive Dijkstra with std::priority_queue (lazy deletion)
std::vector<double> dijkstra_std_priority_queue(const CSRGraph<double>& graph, VertexId source) {
    const std::size_t num_v = graph.num_vertices();
    constexpr double inf = std::numeric_limits<double>::infinity();
    std::vector<double> dist(num_v, inf);
    dist[source] = 0.0;

    using PQElement = std::pair<double, VertexId>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
    pq.push({0.0, source});

    while (!pq.empty()) {
        auto [d_u, u] = pq.top();
        pq.pop();

        if (d_u > dist[u]) {
            continue; // Lazy deletion skip
        }

        auto neighbors = graph.out_neighbors(u);
        auto weights = graph.out_weights(u);
        const std::size_t deg = neighbors.size();

        for (std::size_t i = 0; i < deg; ++i) {
            const VertexId v = neighbors[i];
            const double w = weights[i];
            const double new_d = d_u + w;

            if (new_d < dist[v]) {
                dist[v] = new_d;
                pq.push({new_d, v});
            }
        }
    }

    return dist;
}

} // namespace

// ============================================================================
// 1. SSSP Dijkstra: 4-Ary Heap vs std::priority_queue (DIMACS USA-Road Model)
// ============================================================================

static void BM_Dijkstra_StdPriorityQueue_DIMACSRoad(benchmark::State& state) {
    const auto grid_dim = static_cast<std::size_t>(state.range(0));
    const auto num_v = grid_dim * grid_dim;
    const auto edges = generate_road_network(grid_dim);
    const auto graph =
        CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);

    for (auto _ : state) {
        auto res = dijkstra_std_priority_queue(graph, 0);
        benchmark::DoNotOptimize(res.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * graph.num_edges());
}

static void BM_Dijkstra_4AryHeap_DIMACSRoad(benchmark::State& state) {
    const auto grid_dim = static_cast<std::size_t>(state.range(0));
    const auto num_v = grid_dim * grid_dim;
    const auto edges = generate_road_network(grid_dim);
    const auto graph =
        CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
    ShortestPathEngine<double> engine(graph);

    for (auto _ : state) {
        auto res = engine.dijkstra(0);
        benchmark::DoNotOptimize(res.distances.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * graph.num_edges());
}

BENCHMARK(BM_Dijkstra_StdPriorityQueue_DIMACSRoad)
    ->Arg(50)
    ->Arg(100)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Dijkstra_4AryHeap_DIMACSRoad)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);

// ============================================================================
// 2. Point-to-Point Search: Standard Dijkstra vs ALT Search (DIMACS USA-Road Model)
// ============================================================================

static void BM_P2P_StandardDijkstra_DIMACSRoad(benchmark::State& state) {
    const auto grid_dim = static_cast<std::size_t>(state.range(0));
    const auto num_v = grid_dim * grid_dim;
    const auto edges = generate_road_network(grid_dim);
    const auto graph =
        CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
    ShortestPathEngine<double> engine(graph);

    const VertexId s = 0;
    const VertexId t = static_cast<VertexId>(num_v - 1);

    std::size_t total_settled = 0;

    for (auto _ : state) {
        std::size_t settled = 0;
        auto dist = engine.dijkstra_p2p(s, t, nullptr, &settled);
        total_settled += settled;
        benchmark::DoNotOptimize(dist);
        benchmark::ClobberMemory();
    }

    state.counters["avg_nodes_settled"] = static_cast<double>(total_settled) / state.iterations();
}

static void BM_P2P_ALTSearch_DIMACSRoad(benchmark::State& state) {
    const auto grid_dim = static_cast<std::size_t>(state.range(0));
    const auto num_v = grid_dim * grid_dim;
    const auto edges = generate_road_network(grid_dim);
    const auto graph =
        CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
    ShortestPathEngine<double> engine(graph);
    engine.preprocess_landmarks(16);

    const VertexId s = 0;
    const VertexId t = static_cast<VertexId>(num_v - 1);

    std::size_t total_settled = 0;

    for (auto _ : state) {
        std::size_t settled = 0;
        auto dist = engine.alt_search(s, t, nullptr, &settled);
        total_settled += settled;
        benchmark::DoNotOptimize(dist);
        benchmark::ClobberMemory();
    }

    state.counters["avg_nodes_settled"] = static_cast<double>(total_settled) / state.iterations();
}

BENCHMARK(BM_P2P_StandardDijkstra_DIMACSRoad)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_P2P_ALTSearch_DIMACSRoad)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);

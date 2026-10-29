/**
 * @file bench_max_flow.cpp
 * @brief Google Benchmark suite comparing Dinic's Algorithm and Highest-Label Push-Relabel (HLPR)
 * against Edmonds-Karp augmenting paths baseline across DIMACS (genrmf, wash, ak) and dense
 * networks.
 */

#include "algoat/graph/max_flow.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <utility>
#include <vector>

using namespace algoat::graph;

namespace {

// Edmonds-Karp Baseline (Shortest Augmenting Path via BFS, O(V E^2))
int64_t edmonds_karp_baseline(std::size_t num_v, const std::vector<FlowEdge<int64_t>>& edges,
                              VertexId source, VertexId sink) {
    if (num_v == 0 || source == sink) {
        return 0;
    }

    std::vector<std::vector<std::size_t>> adj(num_v);
    struct Edge {
        VertexId to;
        int64_t cap;
        int64_t flow;
        std::size_t rev;
    };
    std::vector<Edge> edge_list;

    for (const auto& e : edges) {
        const std::size_t id = edge_list.size();
        edge_list.push_back({e.to, e.capacity, 0, id + 1});
        edge_list.push_back({e.from, 0, 0, id});
        adj[e.from].push_back(id);
        adj[e.to].push_back(id + 1);
    }

    int64_t max_flow = 0;
    while (true) {
        std::vector<std::size_t> parent_edge(num_v, std::numeric_limits<std::size_t>::max());
        std::vector<bool> visited(num_v, false);
        std::queue<VertexId> q;

        q.push(source);
        visited[source] = true;

        while (!q.empty() && !visited[sink]) {
            const VertexId u = q.front();
            q.pop();

            for (std::size_t e_idx : adj[u]) {
                const auto& edge = edge_list[e_idx];
                if (!visited[edge.to] && edge.cap > edge.flow) {
                    visited[edge.to] = true;
                    parent_edge[edge.to] = e_idx;
                    q.push(edge.to);
                }
            }
        }

        if (!visited[sink]) {
            break;
        }

        int64_t push = std::numeric_limits<int64_t>::max();
        for (VertexId v = sink; v != source;) {
            std::size_t e_idx = parent_edge[v];
            push = std::min(push, edge_list[e_idx].cap - edge_list[e_idx].flow);
            v = edge_list[edge_list[e_idx].rev].to;
        }

        for (VertexId v = sink; v != source;) {
            std::size_t e_idx = parent_edge[v];
            edge_list[e_idx].flow += push;
            edge_list[edge_list[e_idx].rev].flow -= push;
            v = edge_list[edge_list[e_idx].rev].to;
        }
        max_flow += push;
    }
    return max_flow;
}

// ---------------------------------------------------------------------------
// DIMACS Max-Flow Benchmark Generators: GenRMF, Wash, and AK
// ---------------------------------------------------------------------------

// DIMACS GenRMF: Multiple square frames connected in a layer chain
std::vector<FlowEdge<int64_t>> generate_dimacs_genrmf(std::size_t frame_dim, std::size_t num_frames,
                                                      uint64_t seed = 42) {
    const std::size_t frame_size = frame_dim * frame_dim;
    const std::size_t total_v = frame_size * num_frames + 2;
    const VertexId s = 0;
    const VertexId t = static_cast<VertexId>(total_v - 1);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> inter_cap(1, 100);
    std::uniform_int_distribution<int64_t> intra_cap(1, 1000);

    auto node_id = [frame_size, frame_dim](std::size_t f, std::size_t r, std::size_t c) {
        return static_cast<VertexId>(1 + f * frame_size + r * frame_dim + c);
    };

    std::vector<FlowEdge<int64_t>> edges;

    // Connect source to all nodes in frame 0
    for (std::size_t r = 0; r < frame_dim; ++r) {
        for (std::size_t c = 0; c < frame_dim; ++c) {
            edges.push_back({s, node_id(0, r, c), 5000, 0});
        }
    }

    // Connect intra-frame grid edges and inter-frame transitions
    for (std::size_t f = 0; f < num_frames; ++f) {
        for (std::size_t r = 0; r < frame_dim; ++r) {
            for (std::size_t c = 0; c < frame_dim; ++c) {
                const VertexId u = node_id(f, r, c);
                if (r + 1 < frame_dim) {
                    edges.push_back({u, node_id(f, r + 1, c), intra_cap(rng), 0});
                }
                if (c + 1 < frame_dim) {
                    edges.push_back({u, node_id(f, r, c + 1), intra_cap(rng), 0});
                }
                if (f + 1 < num_frames) {
                    edges.push_back({u, node_id(f + 1, r, c), inter_cap(rng), 0});
                }
            }
        }
    }

    // Connect last frame to sink
    for (std::size_t r = 0; r < frame_dim; ++r) {
        for (std::size_t c = 0; c < frame_dim; ++c) {
            edges.push_back({node_id(num_frames - 1, r, c), t, 5000, 0});
        }
    }

    return edges;
}

// DIMACS Washington (Wash): Line mesh network with high-degree cross-links
std::vector<FlowEdge<int64_t>> generate_dimacs_wash(std::size_t lines, std::size_t line_len,
                                                    uint64_t seed = 42) {
    const std::size_t total_v = lines * line_len + 2;
    const VertexId s = 0;
    const VertexId t = static_cast<VertexId>(total_v - 1);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist_cap(1, 100);

    auto node_id = [line_len](std::size_t l, std::size_t idx) {
        return static_cast<VertexId>(1 + l * line_len + idx);
    };

    std::vector<FlowEdge<int64_t>> edges;
    for (std::size_t l = 0; l < lines; ++l) {
        edges.push_back({s, node_id(l, 0), dist_cap(rng) * 5, 0});
        for (std::size_t i = 0; i + 1 < line_len; ++i) {
            edges.push_back({node_id(l, i), node_id(l, i + 1), dist_cap(rng), 0});
            if (l + 1 < lines) {
                edges.push_back({node_id(l, i), node_id(l + 1, i + 1), dist_cap(rng), 0});
            }
        }
        edges.push_back({node_id(l, line_len - 1), t, dist_cap(rng) * 5, 0});
    }
    return edges;
}

// DIMACS AK: Adversarial pathological diamond and cycle network
std::vector<FlowEdge<int64_t>> generate_dimacs_ak(std::size_t k) {
    const std::size_t total_v = 2 * k + 2;
    const VertexId s = 0;
    const VertexId t = static_cast<VertexId>(total_v - 1);

    std::vector<FlowEdge<int64_t>> edges;
    for (std::size_t i = 1; i <= k; ++i) {
        const auto u = static_cast<VertexId>(i);
        const auto v = static_cast<VertexId>(i + k);
        edges.push_back({s, u, 100, 0});
        edges.push_back({u, v, 1, 0}); // Bottleneck
        edges.push_back({v, t, 100, 0});
        if (i < k) {
            edges.push_back({u, static_cast<VertexId>(i + 1), 50, 0});
            edges.push_back({v, static_cast<VertexId>(i + k + 1), 50, 0});
        }
    }
    return edges;
}

// Dense complete flow network
std::vector<FlowEdge<int64_t>> generate_dense_flow_network(std::size_t num_v, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist_cap(1, 500);

    std::vector<FlowEdge<int64_t>> edges;
    for (VertexId u = 0; u < num_v; ++u) {
        for (VertexId v = u + 1; v < num_v; ++v) {
            edges.push_back({u, v, dist_cap(rng), 0});
        }
    }
    return edges;
}

} // namespace

// ============================================================================
// 1. DIMACS GenRMF: Edmonds-Karp vs Dinic vs Push-Relabel
// ============================================================================

static void BM_MaxFlow_DIMACS_GenRMF_EdmondsKarp(benchmark::State& state) {
    const auto dim = static_cast<std::size_t>(state.range(0));
    const auto edges = generate_dimacs_genrmf(dim, 4);
    const std::size_t num_v = dim * dim * 4 + 2;

    for (auto _ : state) {
        int64_t flow = edmonds_karp_baseline(num_v, edges, 0, num_v - 1);
        benchmark::DoNotOptimize(flow);
    }
}
BENCHMARK(BM_MaxFlow_DIMACS_GenRMF_EdmondsKarp)->Arg(4)->Arg(6);

static void BM_MaxFlow_DIMACS_GenRMF_Dinic(benchmark::State& state) {
    const auto dim = static_cast<std::size_t>(state.range(0));
    const auto edges = generate_dimacs_genrmf(dim, 4);
    const std::size_t num_v = dim * dim * 4 + 2;

    MaxFlowEngine<int64_t> engine(num_v);
    for (const auto& e : edges) {
        engine.add_edge(e.from, e.to, e.capacity);
    }

    for (auto _ : state) {
        auto res = engine.dinic(0, num_v - 1);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_MaxFlow_DIMACS_GenRMF_Dinic)->Arg(4)->Arg(6)->Arg(8);

static void BM_MaxFlow_DIMACS_GenRMF_PushRelabel(benchmark::State& state) {
    const auto dim = static_cast<std::size_t>(state.range(0));
    const auto edges = generate_dimacs_genrmf(dim, 4);
    const std::size_t num_v = dim * dim * 4 + 2;

    MaxFlowEngine<int64_t> engine(num_v);
    for (const auto& e : edges) {
        engine.add_edge(e.from, e.to, e.capacity);
    }

    for (auto _ : state) {
        auto res = engine.push_relabel(0, num_v - 1);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_MaxFlow_DIMACS_GenRMF_PushRelabel)->Arg(4)->Arg(6)->Arg(8);

// ============================================================================
// 2. DIMACS Wash: Dinic vs Push-Relabel
// ============================================================================

static void BM_MaxFlow_DIMACS_Wash_Dinic(benchmark::State& state) {
    const auto lines = static_cast<std::size_t>(state.range(0));
    const auto edges = generate_dimacs_wash(lines, 32);
    const std::size_t num_v = lines * 32 + 2;

    MaxFlowEngine<int64_t> engine(num_v);
    for (const auto& e : edges) {
        engine.add_edge(e.from, e.to, e.capacity);
    }

    for (auto _ : state) {
        auto res = engine.dinic(0, num_v - 1);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_MaxFlow_DIMACS_Wash_Dinic)->Arg(8)->Arg(16);

static void BM_MaxFlow_DIMACS_Wash_PushRelabel(benchmark::State& state) {
    const auto lines = static_cast<std::size_t>(state.range(0));
    const auto edges = generate_dimacs_wash(lines, 32);
    const std::size_t num_v = lines * 32 + 2;

    MaxFlowEngine<int64_t> engine(num_v);
    for (const auto& e : edges) {
        engine.add_edge(e.from, e.to, e.capacity);
    }

    for (auto _ : state) {
        auto res = engine.push_relabel(0, num_v - 1);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_MaxFlow_DIMACS_Wash_PushRelabel)->Arg(8)->Arg(16);

// ============================================================================
// 3. DIMACS AK (Pathological Adversarial): Dinic vs Push-Relabel
// ============================================================================

static void BM_MaxFlow_DIMACS_AK_Dinic(benchmark::State& state) {
    const auto k = static_cast<std::size_t>(state.range(0));
    const auto edges = generate_dimacs_ak(k);
    const std::size_t num_v = 2 * k + 2;

    MaxFlowEngine<int64_t> engine(num_v);
    for (const auto& e : edges) {
        engine.add_edge(e.from, e.to, e.capacity);
    }

    for (auto _ : state) {
        auto res = engine.dinic(0, num_v - 1);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_MaxFlow_DIMACS_AK_Dinic)->Arg(16)->Arg(32);

static void BM_MaxFlow_DIMACS_AK_PushRelabel(benchmark::State& state) {
    const auto k = static_cast<std::size_t>(state.range(0));
    const auto edges = generate_dimacs_ak(k);
    const std::size_t num_v = 2 * k + 2;

    MaxFlowEngine<int64_t> engine(num_v);
    for (const auto& e : edges) {
        engine.add_edge(e.from, e.to, e.capacity);
    }

    for (auto _ : state) {
        auto res = engine.push_relabel(0, num_v - 1);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_MaxFlow_DIMACS_AK_PushRelabel)->Arg(16)->Arg(32);

// ============================================================================
// 4. Dense Networks
// ============================================================================

static void BM_MaxFlow_Dense_Dinic(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto edges = generate_dense_flow_network(num_v);

    MaxFlowEngine<int64_t> engine(num_v);
    for (const auto& e : edges) {
        engine.add_edge(e.from, e.to, e.capacity);
    }

    for (auto _ : state) {
        auto res = engine.dinic(0, num_v - 1);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_MaxFlow_Dense_Dinic)->Arg(32)->Arg(64);

static void BM_MaxFlow_Dense_PushRelabel(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto edges = generate_dense_flow_network(num_v);

    MaxFlowEngine<int64_t> engine(num_v);
    for (const auto& e : edges) {
        engine.add_edge(e.from, e.to, e.capacity);
    }

    for (auto _ : state) {
        auto res = engine.push_relabel(0, num_v - 1);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_MaxFlow_Dense_PushRelabel)->Arg(32)->Arg(64);

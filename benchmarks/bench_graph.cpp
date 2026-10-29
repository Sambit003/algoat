/**
 * @file bench_graph.cpp
 * @brief Google Benchmark suite for CSRGraph vs std::vector<std::vector<VertexId>> traversal.
 */

#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

using namespace algoat::graph;

// Helper to generate an Erdős–Rényi or synthetic power-law graph
static std::vector<std::pair<VertexId, VertexId>>
generate_graph_edges(std::size_t num_v, std::size_t num_e, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<VertexId> dist(0, static_cast<VertexId>(num_v - 1));

    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(num_e);
    for (std::size_t i = 0; i < num_e; ++i) {
        edges.emplace_back(dist(rng), dist(rng));
    }
    return edges;
}

static std::vector<std::vector<VertexId>>
build_vector_of_vectors(std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& edges,
                        bool directed) {
    std::vector<std::vector<VertexId>> adj(num_v);
    for (const auto& [u, v] : edges) {
        if (u < num_v && v < num_v) {
            adj[u].push_back(v);
            if (!directed && u != v) {
                adj[v].push_back(u);
            }
        }
    }
    for (std::size_t u = 0; u < num_v; ++u) {
        std::sort(adj[u].begin(), adj[u].end());
        adj[u].erase(std::unique(adj[u].begin(), adj[u].end()), adj[u].end());
    }
    return adj;
}

// ---------------------------------------------------------------------------
// 1. Neighbor Iteration Traversal Benchmark
// ---------------------------------------------------------------------------

static void BM_NeighborIteration_CSR(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);
    const CSRGraph g = CSRGraph<>::build_from_edges(num_v, edges, /*directed=*/true);

    for (auto _ : state) {
        uint64_t sum = 0;
        for (VertexId u = 0; u < num_v; ++u) {
            for (VertexId v : g.out_neighbors(u)) {
                sum += v;
            }
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * g.num_edges());
    state.counters["edges_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations() * g.num_edges()), benchmark::Counter::kIsRate);
}

static void BM_NeighborIteration_CSR_DirectStreaming(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);
    const CSRGraph g = CSRGraph<>::build_from_edges(num_v, edges, /*directed=*/true);
    const auto* col_idx = g.col_idx_data();
    const auto total_edges = g.num_edges();

    for (auto _ : state) {
        uint64_t sum = 0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : sum)
#endif
        for (std::size_t e = 0; e < total_edges; ++e) {
            sum += col_idx[e];
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * total_edges);
    state.counters["edges_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations() * total_edges), benchmark::Counter::kIsRate);
}

static void BM_NeighborIteration_VectorOfVectors(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);
    const auto adj = build_vector_of_vectors(num_v, edges, /*directed=*/true);

    uint64_t total_edges = 0;
    for (const auto& list : adj) {
        total_edges += list.size();
    }

    for (auto _ : state) {
        uint64_t sum = 0;
        for (VertexId u = 0; u < num_v; ++u) {
            for (VertexId v : adj[u]) {
                sum += v;
            }
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * total_edges);
    state.counters["edges_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations() * total_edges), benchmark::Counter::kIsRate);
}

// ---------------------------------------------------------------------------
// 2. Breadth-First Search (BFS) Traversal Benchmark
// ---------------------------------------------------------------------------

static void BM_BFS_CSR(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);
    const CSRGraph g = CSRGraph<>::build_from_edges(num_v, edges, /*directed=*/true);

    std::vector<uint32_t> dist(num_v);
    std::vector<VertexId> queue(num_v);

    for (auto _ : state) {
        std::fill(dist.begin(), dist.end(), UINT32_MAX);
        std::size_t head = 0;
        std::size_t tail = 0;

        dist[0] = 0;
        queue[tail++] = 0;

        while (head < tail) {
            VertexId u = queue[head++];
            uint32_t next_d = dist[u] + 1;
            for (VertexId v : g.out_neighbors(u)) {
                if (dist[v] == UINT32_MAX) {
                    dist[v] = next_d;
                    queue[tail++] = v;
                }
            }
        }
        benchmark::DoNotOptimize(tail);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * g.num_edges());
}

static void BM_BFS_VectorOfVectors(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);
    const auto adj = build_vector_of_vectors(num_v, edges, /*directed=*/true);

    std::vector<uint32_t> dist(num_v);
    std::vector<VertexId> queue(num_v);

    uint64_t total_edges = 0;
    for (const auto& list : adj) {
        total_edges += list.size();
    }

    for (auto _ : state) {
        std::fill(dist.begin(), dist.end(), UINT32_MAX);
        std::size_t head = 0;
        std::size_t tail = 0;

        dist[0] = 0;
        queue[tail++] = 0;

        while (head < tail) {
            VertexId u = queue[head++];
            uint32_t next_d = dist[u] + 1;
            for (VertexId v : adj[u]) {
                if (dist[v] == UINT32_MAX) {
                    dist[v] = next_d;
                    queue[tail++] = v;
                }
            }
        }
        benchmark::DoNotOptimize(tail);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * total_edges);
}

// ---------------------------------------------------------------------------
// 3. High-Throughput Construction Benchmark
// ---------------------------------------------------------------------------

static void BM_CSRGraph_Construction(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);

    for (auto _ : state) {
        CSRGraph g = CSRGraph<>::build_from_edges(num_v, edges, /*directed=*/true);
        benchmark::DoNotOptimize(g.num_edges());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_e);
    state.counters["edges_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations() * num_e), benchmark::Counter::kIsRate);
}

// ---------------------------------------------------------------------------
// 4. Random Vertex Neighborhood Query Benchmark (Graph Random Walk / Traversal)
// ---------------------------------------------------------------------------

static void BM_RandomVertexQueries_CSR(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);
    const CSRGraph g = CSRGraph<>::build_from_edges(num_v, edges, /*directed=*/true);

    constexpr std::size_t kQueryCount = 100000;
    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<VertexId> dist(0, static_cast<VertexId>(num_v - 1));
    std::vector<VertexId> query_vertices(kQueryCount);
    for (std::size_t i = 0; i < kQueryCount; ++i) {
        query_vertices[i] = dist(rng);
    }

    for (auto _ : state) {
        uint64_t sum = 0;
        for (VertexId u : query_vertices) {
            for (VertexId v : g.out_neighbors(u)) {
                sum += v;
            }
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * kQueryCount * avg_deg);
}

static void BM_RandomVertexQueries_VectorOfVectors(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);
    const auto adj = build_vector_of_vectors(num_v, edges, /*directed=*/true);

    constexpr std::size_t kQueryCount = 100000;
    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<VertexId> dist(0, static_cast<VertexId>(num_v - 1));
    std::vector<VertexId> query_vertices(kQueryCount);
    for (std::size_t i = 0; i < kQueryCount; ++i) {
        query_vertices[i] = dist(rng);
    }

    for (auto _ : state) {
        uint64_t sum = 0;
        for (VertexId u : query_vertices) {
            for (VertexId v : adj[u]) {
                sum += v;
            }
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * kQueryCount * avg_deg);
}

// ---------------------------------------------------------------------------
// 5. Sparse Matrix-Vector Multiplication (SpMV) Kernel
// ---------------------------------------------------------------------------

static void BM_SpMV_CSR(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);
    const CSRGraph g = CSRGraph<>::build_from_edges(num_v, edges, /*directed=*/true);

    std::vector<float> x(num_v, 1.0f);
    std::vector<float> y(num_v, 0.0f);

    const auto* row_ptr = g.row_ptr_data();
    const auto* col_idx = g.col_idx_data();

    for (auto _ : state) {
        for (VertexId u = 0; u < num_v; ++u) {
            float sum = 0.0f;
            const EdgeIndex start = row_ptr[u];
            const EdgeIndex end = row_ptr[u + 1];
            for (EdgeIndex e = start; e < end; ++e) {
                sum += x[col_idx[e]];
            }
            y[u] = sum;
        }
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * g.num_edges());
}

static void BM_SpMV_VectorOfVectors(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto avg_deg = static_cast<std::size_t>(state.range(1));
    const auto num_e = num_v * avg_deg;

    const auto edges = generate_graph_edges(num_v, num_e);
    const auto adj = build_vector_of_vectors(num_v, edges, /*directed=*/true);

    std::vector<float> x(num_v, 1.0f);
    std::vector<float> y(num_v, 0.0f);

    uint64_t total_edges = 0;
    for (const auto& list : adj) {
        total_edges += list.size();
    }

    for (auto _ : state) {
        for (VertexId u = 0; u < num_v; ++u) {
            float sum = 0.0f;
            for (VertexId v : adj[u]) {
                sum += x[v];
            }
            y[u] = sum;
        }
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * total_edges);
}

// ---------------------------------------------------------------------------
// 6. Massive Scale 10M Vertices & 100M Edges Construction
// ---------------------------------------------------------------------------

static void BM_CSRGraph_MassiveScale_Undirected_10M_100M(benchmark::State& state) {
    constexpr std::size_t num_v = 10'000'000;
    // 50M undirected edges yield 100M directed entries in CSR (|V| = 10^7, |E| = 10^8)
    constexpr std::size_t num_e = 50'000'000;

    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(num_e);
    uint64_t state_rng = 123456789ULL;
    for (std::size_t i = 0; i < num_e; ++i) {
        state_rng = state_rng * 6364136223846793005ULL + 1442695040888963407ULL;
        VertexId u = static_cast<VertexId>((state_rng >> 32) % num_v);
        state_rng = state_rng * 6364136223846793005ULL + 1442695040888963407ULL;
        VertexId v = static_cast<VertexId>((state_rng >> 32) % num_v);
        edges.emplace_back(u, v);
    }

    for (auto _ : state) {
        CSRGraph g = CSRGraph<>::build_from_edges(num_v, edges, /*directed=*/false);
        benchmark::DoNotOptimize(g.num_edges());
        benchmark::ClobberMemory();
    }

    constexpr std::size_t total_csr_edges = num_e * 2;
    state.SetItemsProcessed(state.iterations() * total_csr_edges);
    state.counters["edges_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations() * total_csr_edges), benchmark::Counter::kIsRate);
}

static void BM_CSRGraph_MassiveScale_Directed_10M_100M(benchmark::State& state) {
    constexpr std::size_t num_v = 10'000'000;
    constexpr std::size_t num_e = 100'000'000;

    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(num_e);
    uint64_t state_rng = 123456789ULL;
    for (std::size_t i = 0; i < num_e; ++i) {
        state_rng = state_rng * 6364136223846793005ULL + 1442695040888963407ULL;
        VertexId u = static_cast<VertexId>((state_rng >> 32) % num_v);
        state_rng = state_rng * 6364136223846793005ULL + 1442695040888963407ULL;
        VertexId v = static_cast<VertexId>((state_rng >> 32) % num_v);
        edges.emplace_back(u, v);
    }

    for (auto _ : state) {
        CSRGraph g = CSRGraph<>::build_from_edges(num_v, edges, /*directed=*/true);
        benchmark::DoNotOptimize(g.num_edges());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_e);
    state.counters["edges_per_sec"] = benchmark::Counter(
        static_cast<double>(state.iterations() * num_e), benchmark::Counter::kIsRate);
}

// Register Benchmarks: low-degree sparse networks (RoadNet/Web topology: deg 2, 4) and denser
// networks (deg 16)
BENCHMARK(BM_NeighborIteration_CSR)->Args({500000, 2})->Args({500000, 4})->Args({500000, 16});
BENCHMARK(BM_NeighborIteration_CSR_DirectStreaming)
    ->Args({500000, 2})
    ->Args({500000, 4})
    ->Args({500000, 16});
BENCHMARK(BM_NeighborIteration_VectorOfVectors)
    ->Args({500000, 2})
    ->Args({500000, 4})
    ->Args({500000, 16});
BENCHMARK(BM_SpMV_CSR)->Args({500000, 4})->Args({500000, 16});
BENCHMARK(BM_SpMV_VectorOfVectors)->Args({500000, 4})->Args({500000, 16});
BENCHMARK(BM_RandomVertexQueries_CSR)->Args({500000, 4})->Args({500000, 16});
BENCHMARK(BM_RandomVertexQueries_VectorOfVectors)->Args({500000, 4})->Args({500000, 16});
BENCHMARK(BM_BFS_CSR)->Args({500000, 2})->Args({500000, 4})->Args({500000, 16});
BENCHMARK(BM_BFS_VectorOfVectors)->Args({500000, 2})->Args({500000, 4})->Args({500000, 16});
BENCHMARK(BM_CSRGraph_Construction)->Args({100000, 16})->Args({500000, 16});
BENCHMARK(BM_CSRGraph_MassiveScale_Undirected_10M_100M)->Iterations(1);
BENCHMARK(BM_CSRGraph_MassiveScale_Directed_10M_100M)->Iterations(1);

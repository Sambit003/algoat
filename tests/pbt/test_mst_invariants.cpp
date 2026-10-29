#include "algoat/graph/mst.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include <rapidcheck.h>
#include <set>
#include <tuple>
#include <vector>

using namespace algoat::graph;

namespace {

std::vector<std::tuple<VertexId, VertexId, double>>
generate_random_edges(std::size_t num_v,
                      const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges) {
    std::vector<std::tuple<VertexId, VertexId, double>> valid_edges;
    valid_edges.reserve(raw_edges.size());
    for (const auto& [u, v, raw_w] : raw_edges) {
        // Edge weights between [-500.0, 500.0]
        double w = static_cast<double>(static_cast<int32_t>(raw_w % 1001) - 500);
        valid_edges.emplace_back(u % num_v, v % num_v, w);
    }
    return valid_edges;
}

} // namespace

// ============================================================================
// 1. RapidCheck: Serial and Parallel Weight Agreement
// ============================================================================

TEST(MSTPBT, SerialAndParallelWeightExactAgreement) {
    rc::check("Parallel MST weight matches serial Kruskal weight exactly across arbitrary graphs",
              [](std::size_t num_v,
                 const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges,
                 uint8_t raw_threads) {
                  num_v = (num_v % 60) + 1; // 1 to 60 vertices
                  const auto edges = generate_random_edges(num_v, raw_edges);
                  auto g =
                      CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
                  MinimumSpanningTreeEngine<double> engine(g);

                  auto serial_res = engine.compute_serial();
                  std::size_t threads = (raw_threads % 4) + 1; // 1 to 4 threads
                  auto parallel_res = engine.compute_parallel(threads);

                  RC_ASSERT(std::abs(serial_res.total_weight - parallel_res.total_weight) < 1e-6);
                  RC_ASSERT(serial_res.components_count == parallel_res.components_count);
                  RC_ASSERT(serial_res.edges.size() == parallel_res.edges.size());
              });
}

// ============================================================================
// 2. RapidCheck: Acyclic Forest & Component Invariants
// ============================================================================

TEST(MSTPBT, AcyclicForestAndComponentInvariants) {
    rc::check("Returned edge set forms an acyclic forest satisfying |E| = |V| - k",
              [](std::size_t num_v,
                 const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges) {
                  num_v = (num_v % 70) + 1; // 1 to 70 vertices
                  const auto edges = generate_random_edges(num_v, raw_edges);
                  auto g =
                      CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
                  MinimumSpanningTreeEngine<double> engine(g);

                  auto res = engine.compute_parallel(2);

                  // Invariant 1: Forest edge count equals |V| - component_count
                  RC_ASSERT(res.edges.size() + res.components_count == num_v);

                  // Invariant 2: Acyclic verification via independent DSU
                  DisjointSetUnion verify_dsu(num_v);
                  for (const auto& edge : res.edges) {
                      // unite must return true for every edge in a valid acyclic forest
                      RC_ASSERT(verify_dsu.unite(edge.u, edge.v));
                  }
                  RC_ASSERT(verify_dsu.num_components() == res.components_count);
              });
}

// ============================================================================
// 3. Acceptance Criteria: 1,000 Randomized Graphs Verification Suite
// ============================================================================

TEST(MSTPBT, OneThousandRandomGraphsVerification) {
    constexpr int NUM_GRAPHS = 1000;
    std::mt19937_64 rng(20261029);

    std::uniform_int_distribution<std::size_t> v_dist(1, 100);
    std::uniform_real_distribution<double> weight_dist(-1000.0, 1000.0);

    for (int g_idx = 0; g_idx < NUM_GRAPHS; ++g_idx) {
        std::size_t n = v_dist(rng);
        std::uniform_int_distribution<VertexId> vertex_picker(0, static_cast<VertexId>(n - 1));

        std::size_t m = (n <= 1) ? 0 : std::uniform_int_distribution<std::size_t>(0, n * 3)(rng);
        std::vector<std::tuple<VertexId, VertexId, double>> edges;
        edges.reserve(m);

        for (std::size_t i = 0; i < m; ++i) {
            VertexId u = vertex_picker(rng);
            VertexId v = vertex_picker(rng);
            edges.emplace_back(u, v, weight_dist(rng));
        }

        auto g = CSRGraph<double>::build_from_weighted_edges(n, edges, /*directed=*/false);
        MinimumSpanningTreeEngine<double> engine(g);

        auto serial_res = engine.compute_serial();
        auto parallel_res = engine.compute_parallel(4);

        ASSERT_NEAR(parallel_res.total_weight, serial_res.total_weight, 1e-6)
            << "Mismatch on graph " << g_idx << " with V=" << n << ", E=" << m;
        ASSERT_EQ(parallel_res.components_count, serial_res.components_count);
        ASSERT_EQ(parallel_res.edges.size(), serial_res.edges.size());

        // Verify cycle-free
        DisjointSetUnion dsu(n);
        for (const auto& edge : parallel_res.edges) {
            ASSERT_TRUE(dsu.unite(edge.u, edge.v));
        }
        ASSERT_EQ(dsu.num_components(), parallel_res.components_count);
    }
}

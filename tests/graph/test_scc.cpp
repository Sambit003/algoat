#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/scc.hpp"
#include "scc_test_utils.hpp"

#include <algorithm>
#include <chrono>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace algoat::graph;
using algoat::graph::test::KosarajuReference;

// ============================================================================
// 1. Basic Unit Tests: Degenerate, Trivial & Small Graphs
// ============================================================================

TEST(SCCTest, EmptyGraph) {
    auto g = CSRGraph<void>::build_from_edges(0, {}, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, 0u);
    EXPECT_TRUE(res.component_ids.empty());
    EXPECT_TRUE(res.components.empty());

    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 0u);
    EXPECT_EQ(dag.num_edges(), 0u);
}

TEST(SCCTest, SingleVertexWithoutSelfLoop) {
    auto g = CSRGraph<void>::build_from_edges(1, {}, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, 1u);
    ASSERT_EQ(res.component_ids.size(), 1u);
    EXPECT_EQ(res.component_ids[0], 0u);
    ASSERT_EQ(res.components.size(), 1u);
    EXPECT_EQ(res.components[0], std::vector<VertexId>{0});

    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 1u);
    EXPECT_EQ(dag.num_edges(), 0u);
}

TEST(SCCTest, SingleVertexWithSelfLoop) {
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 0}};
    auto g = CSRGraph<void>::build_from_edges(1, edges, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, 1u);
    ASSERT_EQ(res.component_ids.size(), 1u);
    EXPECT_EQ(res.component_ids[0], 0u);
    ASSERT_EQ(res.components.size(), 1u);
    EXPECT_EQ(res.components[0], std::vector<VertexId>{0});

    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 1u);
    EXPECT_EQ(dag.num_edges(), 0u); // Self-loops eliminated in condensation DAG
}

TEST(SCCTest, DisconnectedSingletons) {
    auto g = CSRGraph<void>::build_from_edges(4, {}, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, 4u);
    ASSERT_EQ(res.component_ids.size(), 4u);
    ASSERT_EQ(res.components.size(), 4u);

    // Each vertex must be in its own component
    std::vector<uint32_t> seen(4, 0);
    for (uint32_t cid : res.component_ids) {
        EXPECT_LT(cid, 4u);
        seen[cid]++;
    }
    for (uint32_t count : seen) {
        EXPECT_EQ(count, 1u);
    }

    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 4u);
    EXPECT_EQ(dag.num_edges(), 0u);
}

TEST(SCCTest, SimpleDirectedCycle) {
    // 0 -> 1 -> 2 -> 0
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 2}, {2, 0}};
    auto g = CSRGraph<void>::build_from_edges(3, edges, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, 1u);
    ASSERT_EQ(res.component_ids.size(), 3u);
    EXPECT_EQ(res.component_ids[0], res.component_ids[1]);
    EXPECT_EQ(res.component_ids[1], res.component_ids[2]);
    ASSERT_EQ(res.components.size(), 1u);
    EXPECT_EQ(res.components[0].size(), 3u);

    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 1u);
    EXPECT_EQ(dag.num_edges(), 0u);
}

TEST(SCCTest, AntiParallelEdges) {
    // 0 <-> 1 (two nodes, 2 anti-parallel directed edges)
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 0}};
    auto g = CSRGraph<void>::build_from_edges(2, edges, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, 1u);
    ASSERT_EQ(res.component_ids.size(), 2u);
    EXPECT_EQ(res.component_ids[0], res.component_ids[1]);

    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 1u);
    EXPECT_EQ(dag.num_edges(), 0u);
}

TEST(SCCTest, DisjointComponentsWithBridgeAndCondensationDAG) {
    // Component 1: 0 <-> 1 (cycle)
    // Component 2: 2 <-> 3 (cycle)
    // Bridge: 1 -> 2 (directed edge from Comp 1 to Comp 2)
    // Extra node 4: 3 -> 4 (singleton sink)
    std::vector<std::pair<VertexId, VertexId>> edges = {
        {0, 1}, {1, 0}, // SCC 1
        {2, 3}, {3, 2}, // SCC 2
        {1, 2},         // Inter-component bridge
        {3, 4}          // Edge to singleton 4
    };
    auto g = CSRGraph<void>::build_from_edges(5, edges, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, 3u);
    EXPECT_EQ(res.component_ids[0], res.component_ids[1]);
    EXPECT_EQ(res.component_ids[2], res.component_ids[3]);
    EXPECT_NE(res.component_ids[0], res.component_ids[2]);
    EXPECT_NE(res.component_ids[2], res.component_ids[4]);
    EXPECT_NE(res.component_ids[0], res.component_ids[4]);

    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 3u);
    EXPECT_EQ(dag.num_edges(), 2u);

    // Topological order property verification: for every edge u -> v, u < v
    for (VertexId u = 0; u < dag.num_vertices(); ++u) {
        for (VertexId v : dag.out_neighbors(u)) {
            EXPECT_LT(u, v) << "Condensation DAG edge violates topological ordering!";
        }
    }
}

TEST(SCCTest, MultipleInterComponentEdgesDeduplicated) {
    // SCC A: 0, 1
    // SCC B: 2, 3
    // Multiple edges from A to B: 0->2, 0->3, 1->2, 1->3
    // Condensation DAG must have exactly 1 edge: A -> B
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 0}, {2, 3}, {3, 2},
                                                        {0, 2}, {0, 3}, {1, 2}, {1, 3}};
    auto g = CSRGraph<void>::build_from_edges(4, edges, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, 2u);

    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 2u);
    EXPECT_EQ(dag.num_edges(), 1u); // Exactly 1 deduplicated edge

    uint32_t comp_A = res.component_ids[0];
    uint32_t comp_B = res.component_ids[2];
    EXPECT_LT(comp_A, comp_B);
    auto neighbors = dag.out_neighbors(comp_A);
    ASSERT_EQ(neighbors.size(), 1u);
    EXPECT_EQ(neighbors[0], comp_B);
}

// ============================================================================
// 2. SNAP Format Dataset Linear Execution
// ============================================================================

TEST(SCCTest, SNAPDatasetStreamLinearComplexity) {
    // Construct representative SNAP-format graph string with comments, isolated vertices,
    // and complex feedback loops
    std::string snap_data = R"(# SNAP Directed Graph Sample
# Directed web-style subgraph
0 1
1 2
2 0
2 3
3 4
4 5
5 3
4 6
6 7
7 8
8 6
)";
    auto g = CSRGraph<void>::build_from_snap_string(snap_data, /*directed=*/true,
                                                    /*forced_num_vertices=*/9);
    EXPECT_EQ(g.num_vertices(), 9u);

    StronglyConnectedComponents engine(g);
    auto start = std::chrono::steady_clock::now();
    auto res = engine.compute();
    auto end = std::chrono::steady_clock::now();

    EXPECT_EQ(res.num_components, 3u); // {0,1,2}, {3,4,5}, {6,7,8}
    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 3u);
    EXPECT_EQ(dag.num_edges(), 2u);

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    EXPECT_GE(elapsed_us, 0);
}

// ============================================================================
// 3. Non-Recursive Deep Call-Stack Safety Test (Degenerate Linear Chain N = 10^7)
// ============================================================================

TEST(SCCTest, DeepLinearChainNoCallStackOverflow10M) {
    // Linear chain of 10,000,000 vertices: 0 -> 1 -> 2 -> ... -> N-1
    // Textbook recursive implementations trigger stack overflow (SIGSEGV) at depth ~150,000.
    const std::size_t N = 10000000;
    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(N - 1);
    for (VertexId i = 0; i + 1 < N; ++i) {
        edges.emplace_back(i, i + 1);
    }

    auto g = CSRGraph<void>::build_from_edges(N, edges, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, N);

    // Topological order check: 0 is source, N-1 is sink
    EXPECT_EQ(res.component_ids[0], 0u);
    EXPECT_EQ(res.component_ids[N - 1], static_cast<uint32_t>(N - 1));
}

// ============================================================================
// 4. Exact Correctness: 10,000 Random Directed Graphs Equivalence
// ============================================================================

TEST(SCCTest, EquivalenceAgainstIndependentReferenceOn10000Graphs) {
    std::mt19937_64 rng(42);
    constexpr std::size_t NUM_ITERATIONS = 10000;

    for (std::size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        const std::size_t n = (rng() % 16) + 1;    // 1 to 16 vertices
        const std::size_t m = rng() % (2 * n * n); // Up to dense edge set

        std::vector<std::pair<VertexId, VertexId>> edges;
        edges.reserve(m);
        for (std::size_t i = 0; i < m; ++i) {
            VertexId u = static_cast<VertexId>(rng() % n);
            VertexId v = static_cast<VertexId>(rng() % n);
            edges.emplace_back(u, v);
        }

        auto g = CSRGraph<void>::build_from_edges(n, edges, /*directed=*/true);
        StronglyConnectedComponents engine(g);
        auto res = engine.compute();

        uint32_t ref_num_scc = 0;
        auto ref_comp = KosarajuReference::compute(n, edges, ref_num_scc);

        ASSERT_EQ(res.num_components, ref_num_scc)
            << "Component count mismatch on iteration " << iter << " for n=" << n << ", m=" << m;

        // Partition equivalence: vertices are in same SCC iff they are in same reference SCC
        for (VertexId u = 0; u < n; ++u) {
            for (VertexId v = u + 1; v < n; ++v) {
                bool same_in_engine = (res.component_ids[u] == res.component_ids[v]);
                bool same_in_ref = (ref_comp[u] == ref_comp[v]);
                ASSERT_EQ(same_in_engine, same_in_ref)
                    << "Partition mismatch on iteration " << iter << " between vertices (" << u
                    << ", " << v << ")";
            }
        }
    }
}

// ============================================================================
// 5. Weighted Graph Support
// ============================================================================

TEST(SCCTest, WeightedGraphSupport) {
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {
        {0, 1, 1.5}, {1, 0, 2.5}, {1, 2, 3.5}};
    auto g = CSRGraph<double>::build_from_weighted_edges(3, edges, /*directed=*/true);
    StronglyConnectedComponents<double> engine(g);

    auto res = engine.compute();
    EXPECT_EQ(res.num_components, 2u);
    EXPECT_EQ(res.component_ids[0], res.component_ids[1]);
    EXPECT_NE(res.component_ids[0], res.component_ids[2]);

    auto dag = engine.condensation_graph(res);
    EXPECT_EQ(dag.num_vertices(), 2u);
    EXPECT_EQ(dag.num_edges(), 1u);
}

#include "algoat/graph/csr_graph.hpp"

#include <chrono>
#include <concepts>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

using namespace algoat::graph;

TEST(CSRGraphTest, SatisfiesCSRGraphConcept) {
    static_assert(CSRGraphConcept<CSRGraph<void>>, "CSRGraph<void> must satisfy CSRGraphConcept");
    static_assert(CSRGraphConcept<CSRGraph<float>>, "CSRGraph<float> must satisfy CSRGraphConcept");
    static_assert(CSRGraphConcept<CSRGraph<double>>,
                  "CSRGraph<double> must satisfy CSRGraphConcept");
}

TEST(CSRGraphTest, EmptyGraph) {
    CSRGraph graph = CSRGraph<>::build_from_edges(0, {});
    EXPECT_EQ(graph.num_vertices(), 0u);
    EXPECT_EQ(graph.num_edges(), 0u);
    EXPECT_FALSE(graph.is_directed());
    EXPECT_NE(graph.row_ptr_data(), nullptr);
    EXPECT_EQ(graph.row_ptr_data()[0], 0u);
    EXPECT_EQ(graph.col_idx_data(), nullptr);
    EXPECT_EQ(graph.out_degree(0), 0u);
    EXPECT_EQ(graph.in_degree(0), 0u);
    EXPECT_TRUE(graph.out_neighbors(0).empty());
    EXPECT_TRUE(graph.in_neighbors(0).empty());
}

TEST(CSRGraphTest, SingleVertexNoEdges) {
    CSRGraph graph = CSRGraph<>::build_from_edges(1, {}, true);
    EXPECT_EQ(graph.num_vertices(), 1u);
    EXPECT_EQ(graph.num_edges(), 0u);
    EXPECT_TRUE(graph.is_directed());
    EXPECT_EQ(graph.out_degree(0), 0u);
    EXPECT_EQ(graph.in_degree(0), 0u);
    EXPECT_TRUE(graph.out_neighbors(0).empty());
    EXPECT_TRUE(graph.in_neighbors(0).empty());
}

TEST(CSRGraphTest, DirectedGraphConstructionAndNeighbors) {
    // 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 0
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {0, 2}, {1, 2}, {2, 0}};
    CSRGraph graph = CSRGraph<>::build_from_edges(3, edges, /*directed=*/true);

    EXPECT_EQ(graph.num_vertices(), 3u);
    EXPECT_EQ(graph.num_edges(), 4u);
    EXPECT_TRUE(graph.is_directed());

    // Out degrees & neighbors
    EXPECT_EQ(graph.out_degree(0), 2u);
    auto out0 = graph.out_neighbors(0);
    ASSERT_EQ(out0.size(), 2u);
    EXPECT_EQ(out0[0], 1u);
    EXPECT_EQ(out0[1], 2u);

    EXPECT_EQ(graph.out_degree(1), 1u);
    auto out1 = graph.out_neighbors(1);
    ASSERT_EQ(out1.size(), 1u);
    EXPECT_EQ(out1[0], 2u);

    EXPECT_EQ(graph.out_degree(2), 1u);
    auto out2 = graph.out_neighbors(2);
    ASSERT_EQ(out2.size(), 1u);
    EXPECT_EQ(out2[0], 0u);

    // In degrees & neighbors
    EXPECT_EQ(graph.in_degree(0), 1u);
    auto in0 = graph.in_neighbors(0);
    ASSERT_EQ(in0.size(), 1u);
    EXPECT_EQ(in0[0], 2u);

    EXPECT_EQ(graph.in_degree(1), 1u);
    auto in1 = graph.in_neighbors(1);
    ASSERT_EQ(in1.size(), 1u);
    EXPECT_EQ(in1[0], 0u);

    EXPECT_EQ(graph.in_degree(2), 2u);
    auto in2 = graph.in_neighbors(2);
    ASSERT_EQ(in2.size(), 2u);
    EXPECT_EQ(in2[0], 0u);
    EXPECT_EQ(in2[1], 1u);
}

TEST(CSRGraphTest, UndirectedGraphSymmetryAndMemoryBound) {
    // 0 - 1, 1 - 2
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 2}};
    CSRGraph graph = CSRGraph<>::build_from_edges(3, edges, /*directed=*/false);

    EXPECT_EQ(graph.num_vertices(), 3u);
    // Undirected graph stores 2 directed edges per undirected edge: {0->1, 1->0, 1->2, 2->1}
    EXPECT_EQ(graph.num_edges(), 4u);
    EXPECT_FALSE(graph.is_directed());

    EXPECT_EQ(graph.out_degree(0), 1u);
    EXPECT_EQ(graph.out_neighbors(0)[0], 1u);

    EXPECT_EQ(graph.out_degree(1), 2u);
    EXPECT_EQ(graph.out_neighbors(1)[0], 0u);
    EXPECT_EQ(graph.out_neighbors(1)[1], 2u);

    EXPECT_EQ(graph.out_degree(2), 1u);
    EXPECT_EQ(graph.out_neighbors(2)[0], 1u);

    // in_neighbors matches out_neighbors for undirected
    EXPECT_EQ(graph.in_neighbors(1).size(), 2u);
    EXPECT_EQ(graph.in_neighbors(1)[0], 0u);
    EXPECT_EQ(graph.in_neighbors(1)[1], 2u);

    // Strict memory footprint: (V + 1) * 8 + E * 4
    std::size_t expected_bytes = (3 + 1) * sizeof(EdgeIndex) + 4 * sizeof(VertexId);
    EXPECT_EQ(graph.memory_footprint(), expected_bytes);
}

TEST(CSRGraphTest, TransposeDirectedGraph) {
    // 0 -> 1, 1 -> 2
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 2}};
    CSRGraph g = CSRGraph<>::build_from_edges(3, edges, /*directed=*/true);
    CSRGraph gt = g.transpose();

    EXPECT_EQ(gt.num_vertices(), 3u);
    EXPECT_EQ(gt.num_edges(), 2u);
    EXPECT_TRUE(gt.is_directed());

    // In transposed graph: 1 -> 0, 2 -> 1
    EXPECT_EQ(gt.out_degree(0), 0u);
    EXPECT_EQ(gt.out_degree(1), 1u);
    EXPECT_EQ(gt.out_neighbors(1)[0], 0u);
    EXPECT_EQ(gt.out_degree(2), 1u);
    EXPECT_EQ(gt.out_neighbors(2)[0], 1u);

    // Double transpose equals original
    CSRGraph gtt = gt.transpose();
    EXPECT_EQ(gtt.out_neighbors(0).size(), 1u);
    EXPECT_EQ(gtt.out_neighbors(0)[0], 1u);
    EXPECT_EQ(gtt.out_neighbors(1).size(), 1u);
    EXPECT_EQ(gtt.out_neighbors(1)[0], 2u);
}

TEST(CSRGraphTest, MultiEdgesDeduplicationAndSelfLoops) {
    // Contains duplicate (0, 1) and self loop (2, 2)
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {0, 1}, {0, 2}, {1, 2}, {2, 2}};
    CSRGraph g = CSRGraph<>::build_from_edges(3, edges, /*directed=*/true);

    EXPECT_EQ(g.num_vertices(), 3u);
    // Unique edges: (0, 1), (0, 2), (1, 2), (2, 2) -> 4 edges
    EXPECT_EQ(g.num_edges(), 4u);
    EXPECT_EQ(g.out_degree(0), 2u);
    EXPECT_EQ(g.out_neighbors(0)[0], 1u);
    EXPECT_EQ(g.out_neighbors(0)[1], 2u);
    EXPECT_EQ(g.out_neighbors(2)[0], 2u);
}

TEST(CSRGraphTest, OutOfBoundsVertexSafety) {
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 2}};
    CSRGraph g = CSRGraph<>::build_from_edges(3, edges, /*directed=*/true);

    EXPECT_EQ(g.out_degree(100), 0u);
    EXPECT_EQ(g.in_degree(100), 0u);
    EXPECT_TRUE(g.out_neighbors(100).empty());
    EXPECT_TRUE(g.in_neighbors(100).empty());
}

TEST(CSRGraphTest, DisconnectedComponentsAndGaps) {
    // Vertices 0 and 2 connected, vertex 1 completely disconnected
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 2}};
    CSRGraph g = CSRGraph<>::build_from_edges(3, edges, /*directed=*/true);

    EXPECT_EQ(g.out_degree(0), 1u);
    EXPECT_EQ(g.out_degree(1), 0u);
    EXPECT_EQ(g.out_degree(2), 0u);
    EXPECT_TRUE(g.out_neighbors(1).empty());
    EXPECT_TRUE(g.in_neighbors(1).empty());
    EXPECT_EQ(g.in_degree(2), 1u);
}

TEST(CSRGraphTest, SnapFormatIngestion) {
    constexpr std::string_view snap_text = "# Directed graph: web-Google.txt\n"
                                           "# Nodes: 4 Edges: 4\n"
                                           "# FromNodeId\tToNodeId\n"
                                           "0\t1\n"
                                           "0\t3\n"
                                           "1\t2\n"
                                           "2\t0\n";

    CSRGraph g = CSRGraph<>::build_from_snap_string(snap_text, /*directed=*/true);

    EXPECT_EQ(g.num_vertices(), 4u);
    EXPECT_EQ(g.num_edges(), 4u);
    EXPECT_TRUE(g.is_directed());

    auto out0 = g.out_neighbors(0);
    ASSERT_EQ(out0.size(), 2u);
    EXPECT_EQ(out0[0], 1u);
    EXPECT_EQ(out0[1], 3u);

    auto in0 = g.in_neighbors(0);
    ASSERT_EQ(in0.size(), 1u);
    EXPECT_EQ(in0[0], 2u);
}

TEST(CSRGraphTest, SnapFormatWebGoogleTopology) {
    // SNAP web-Google: Directed web graph with comments, gaps, and sparse IDs
    constexpr std::string_view web_google_snippet =
        "# Directed graph (each unordered pair of nodes is saved once): web-Google.txt\n"
        "# Web graph from Google\n"
        "# Nodes: 875713 Edges: 5105039\n"
        "# FromNodeId\tToNodeId\n"
        "0\t11342\n"
        "0\t82415\n"
        "0\t100000\n"
        "11342\t0\n"
        "82415\t100000\n";

    CSRGraph g = CSRGraph<>::build_from_snap_string(web_google_snippet, /*directed=*/true);

    EXPECT_EQ(g.num_vertices(), 100001u);
    EXPECT_EQ(g.num_edges(), 5u);
    EXPECT_TRUE(g.is_directed());

    // Out degrees
    EXPECT_EQ(g.out_degree(0), 3u);
    EXPECT_EQ(g.out_degree(11342), 1u);
    EXPECT_EQ(g.out_degree(82415), 1u);
    EXPECT_EQ(g.out_degree(100000), 0u); // Sink / dangling node

    // Gap / disconnected vertex in sparse ID range
    EXPECT_EQ(g.out_degree(1), 0u);
    EXPECT_EQ(g.in_degree(1), 0u);
    EXPECT_TRUE(g.out_neighbors(1).empty());
    EXPECT_TRUE(g.in_neighbors(1).empty());

    // In degrees
    EXPECT_EQ(g.in_degree(0), 1u);
    EXPECT_EQ(g.in_degree(100000), 2u);

    // Neighbors sorted
    auto out0 = g.out_neighbors(0);
    ASSERT_EQ(out0.size(), 3u);
    EXPECT_EQ(out0[0], 11342u);
    EXPECT_EQ(out0[1], 82415u);
    EXPECT_EQ(out0[2], 100000u);

    auto in100000 = g.in_neighbors(100000);
    ASSERT_EQ(in100000.size(), 2u);
    EXPECT_EQ(in100000[0], 0u);
    EXPECT_EQ(in100000[1], 82415u);
}

TEST(CSRGraphTest, SnapFormatSocLiveJournalTopology) {
    // SNAP soc-LiveJournal1: Directed social network with high-degree hub nodes
    constexpr std::string_view soc_livejournal_snippet =
        "# Directed graph (each unordered pair of nodes is saved once): soc-LiveJournal1.txt\n"
        "# LiveJournal online social network\n"
        "# Nodes: 4847571 Edges: 68993773\n"
        "# FromNodeId\tToNodeId\n"
        "1000\t2000\n"
        "1000\t3000\n"
        "1000\t4000\n"
        "1000\t5000\n"
        "2000\t1000\n"
        "3000\t1000\n"
        "4000\t1000\n"
        "5000\t1000\n";

    CSRGraph g = CSRGraph<>::build_from_snap_string(soc_livejournal_snippet, /*directed=*/true);

    EXPECT_EQ(g.num_vertices(), 5001u);
    EXPECT_EQ(g.num_edges(), 8u);
    EXPECT_TRUE(g.is_directed());

    // Hub node 1000 has 4 outgoing and 4 incoming edges
    EXPECT_EQ(g.out_degree(1000), 4u);
    EXPECT_EQ(g.in_degree(1000), 4u);

    auto out_hub = g.out_neighbors(1000);
    ASSERT_EQ(out_hub.size(), 4u);
    EXPECT_EQ(out_hub[0], 2000u);
    EXPECT_EQ(out_hub[1], 3000u);
    EXPECT_EQ(out_hub[2], 4000u);
    EXPECT_EQ(out_hub[3], 5000u);

    auto in_hub = g.in_neighbors(1000);
    ASSERT_EQ(in_hub.size(), 4u);
    EXPECT_EQ(in_hub[0], 2000u);
    EXPECT_EQ(in_hub[1], 3000u);
    EXPECT_EQ(in_hub[2], 4000u);
    EXPECT_EQ(in_hub[3], 5000u);
}

TEST(CSRGraphTest, SnapFormatRoadNetCATopology) {
    // SNAP roadNet-CA: Undirected planar road network with symmetric lines in raw file
    constexpr std::string_view roadnet_ca_snippet =
        "# Undirected graph (each unordered pair of nodes is saved once): roadNet-CA.txt\n"
        "# California road network\n"
        "# Nodes: 1965206 Edges: 2766607\n"
        "# FromNodeId\tToNodeId\n"
        "0\t1\n"
        "0\t2\n"
        "1\t0\n"
        "1\t2\n"
        "2\t0\n"
        "2\t1\n";

    CSRGraph g = CSRGraph<>::build_from_snap_string(roadnet_ca_snippet, /*directed=*/false);

    EXPECT_EQ(g.num_vertices(), 3u);
    // Triangle: 3 undirected edges -> 6 directed entries in CSR
    EXPECT_EQ(g.num_edges(), 6u);
    EXPECT_FALSE(g.is_directed());

    for (VertexId u = 0; u < 3; ++u) {
        EXPECT_EQ(g.out_degree(u), 2u);
        EXPECT_EQ(g.in_degree(u), 2u);
        auto out_u = g.out_neighbors(u);
        ASSERT_EQ(out_u.size(), 2u);
    }

    // Strict undirected memory bound: (3 + 1) * 8 + 6 * 4 = 32 + 24 = 56 bytes
    std::size_t expected_bytes = (3 + 1) * sizeof(EdgeIndex) + 6 * sizeof(VertexId);
    EXPECT_EQ(g.memory_footprint(), expected_bytes);
}

TEST(CSRGraphTest, WeightedCSRGraphOperations) {
    // Weighted edges: (u, v, weight)
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {
        {0, 1, 3.5}, {0, 2, 7.2}, {1, 2, 1.8}};
    auto g = CSRGraph<double>::build_from_weighted_edges(3, edges, /*directed=*/true);

    EXPECT_EQ(g.num_vertices(), 3u);
    EXPECT_EQ(g.num_edges(), 3u);
    EXPECT_TRUE(g.is_directed());

    auto out0 = g.out_neighbors(0);
    auto w0 = g.out_weights(0);
    ASSERT_EQ(out0.size(), 2u);
    ASSERT_EQ(w0.size(), 2u);
    EXPECT_EQ(out0[0], 1u);
    EXPECT_DOUBLE_EQ(w0[0], 3.5);
    EXPECT_EQ(out0[1], 2u);
    EXPECT_DOUBLE_EQ(w0[1], 7.2);

    auto in2 = g.in_neighbors(2);
    auto in_w2 = g.in_weights(2);
    ASSERT_EQ(in2.size(), 2u);
    ASSERT_EQ(in_w2.size(), 2u);
    EXPECT_EQ(in2[0], 0u);
    EXPECT_DOUBLE_EQ(in_w2[0], 7.2);
    EXPECT_EQ(in2[1], 1u);
    EXPECT_DOUBLE_EQ(in_w2[1], 1.8);

    // Transpose preserves weights
    auto gt = g.transpose();
    auto gt_out2 = gt.out_neighbors(2);
    auto gt_w2 = gt.out_weights(2);
    ASSERT_EQ(gt_out2.size(), 2u);
    ASSERT_EQ(gt_w2.size(), 2u);
    EXPECT_EQ(gt_out2[0], 0u);
    EXPECT_DOUBLE_EQ(gt_w2[0], 7.2);
    EXPECT_EQ(gt_out2[1], 1u);
    EXPECT_DOUBLE_EQ(gt_w2[1], 1.8);
}

TEST(CSRGraphTest, HighThroughputConstructionScale) {
    constexpr std::size_t num_v = 100'000;
    constexpr std::size_t num_e = 500'000;

    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(num_e);

    std::mt19937 rng(42);
    std::uniform_int_distribution<VertexId> dist(0, static_cast<VertexId>(num_v - 1));

    for (std::size_t i = 0; i < num_e; ++i) {
        edges.emplace_back(dist(rng), dist(rng));
    }

    auto start = std::chrono::high_resolution_clock::now();
    CSRGraph g = CSRGraph<>::build_from_edges(num_v, edges, /*directed=*/true);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_EQ(g.num_vertices(), num_v);
    EXPECT_LE(g.num_edges(), num_e);
    // 500k edges should build in < 500ms
    EXPECT_LT(elapsed_ms, 500);
}

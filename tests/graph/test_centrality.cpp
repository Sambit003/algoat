#include "algoat/graph/centrality.hpp"
#include "algoat/graph/csr_graph.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace algoat::graph;

TEST(CentralityTest, EmptyGraph) {
    auto g = CSRGraph<void>::build_from_edges(0, {}, /*directed=*/true);
    auto pr = pagerank(g);
    EXPECT_TRUE(pr.scores.empty());
    EXPECT_EQ(pr.iterations_converged, 0u);
    EXPECT_DOUBLE_EQ(pr.final_l1_delta, 0.0);

    auto bc = betweenness_centrality(g);
    EXPECT_TRUE(bc.betweenness.empty());
    EXPECT_TRUE(bc.degree_centrality.empty());
}

TEST(CentralityTest, SingleVertexNoEdges) {
    auto g = CSRGraph<void>::build_from_edges(1, {}, /*directed=*/true);
    auto pr = pagerank(g);
    ASSERT_EQ(pr.scores.size(), 1u);
    EXPECT_DOUBLE_EQ(pr.scores[0], 1.0);

    auto bc = betweenness_centrality(g);
    ASSERT_EQ(bc.betweenness.size(), 1u);
    EXPECT_DOUBLE_EQ(bc.betweenness[0], 0.0);
    ASSERT_EQ(bc.degree_centrality.size(), 1u);
    EXPECT_DOUBLE_EQ(bc.degree_centrality[0], 0.0);
}

TEST(CentralityTest, PageRankDirectedCycle) {
    // 0 -> 1 -> 2 -> 0
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 2}, {2, 0}};
    auto g = CSRGraph<void>::build_from_edges(3, edges, /*directed=*/true);

    PageRankConfig config;
    config.tolerance = 1e-12;
    config.max_iterations = 200;
    auto pr = pagerank(g, config);

    ASSERT_EQ(pr.scores.size(), 3u);
    EXPECT_GT(pr.iterations_converged, 0u);
    EXPECT_LT(pr.final_l1_delta, config.tolerance);

    // Sum invariant sum(p_i) == 1.0
    double sum = std::accumulate(pr.scores.begin(), pr.scores.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-9);

    // By symmetry, all scores should be identical (1/3)
    EXPECT_NEAR(pr.scores[0], 1.0 / 3.0, 1e-6);
    EXPECT_NEAR(pr.scores[1], 1.0 / 3.0, 1e-6);
    EXPECT_NEAR(pr.scores[2], 1.0 / 3.0, 1e-6);
}

TEST(CentralityTest, PageRankDanglingNodeMassRedistribution) {
    // 0 -> 1, 1 has no outgoing edges (dangling node)
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}};
    auto g = CSRGraph<void>::build_from_edges(2, edges, /*directed=*/true);

    PageRankConfig config;
    config.damping_factor = 0.85;
    config.tolerance = 1e-12;
    config.max_iterations = 200;
    auto pr = pagerank(g, config);

    ASSERT_EQ(pr.scores.size(), 2u);
    double sum = std::accumulate(pr.scores.begin(), pr.scores.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-9);

    // Node 1 receives links from Node 0 plus uniform teleportation, so p[1] > p[0]
    EXPECT_GT(pr.scores[1], pr.scores[0]);
}

TEST(CentralityTest, PageRankDisconnectedComponents) {
    // 0 <-> 1 and 2 <-> 3
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 0}, {2, 3}, {3, 2}};
    auto g = CSRGraph<void>::build_from_edges(4, edges, /*directed=*/true);

    auto pr = pagerank(g);
    ASSERT_EQ(pr.scores.size(), 4u);
    double sum = std::accumulate(pr.scores.begin(), pr.scores.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-9);

    for (double score : pr.scores) {
        EXPECT_NEAR(score, 0.25, 1e-5);
    }
}

TEST(CentralityTest, BrandesBetweennessPath3) {
    // Undirected path P3: 0 - 1 - 2
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 2}};
    auto g = CSRGraph<void>::build_from_edges(3, edges, /*directed=*/false);

    auto bc = betweenness_centrality(g);
    ASSERT_EQ(bc.betweenness.size(), 3u);
    ASSERT_EQ(bc.degree_centrality.size(), 3u);

    // Betweenness on P3: only node 1 is an intermediate node (between 0 and 2)
    EXPECT_NEAR(bc.betweenness[0], 0.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[1], 1.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[2], 0.0, 1e-9);

    // Degree centrality normalized by N - 1 = 2:
    // deg(0)=1 -> 0.5, deg(1)=2 -> 1.0, deg(2)=1 -> 0.5
    EXPECT_NEAR(bc.degree_centrality[0], 0.5, 1e-9);
    EXPECT_NEAR(bc.degree_centrality[1], 1.0, 1e-9);
    EXPECT_NEAR(bc.degree_centrality[2], 0.5, 1e-9);
}

TEST(CentralityTest, BrandesBetweennessPath4) {
    // Undirected path P4: 0 - 1 - 2 - 3
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 2}, {2, 3}};
    auto g = CSRGraph<void>::build_from_edges(4, edges, /*directed=*/false);

    auto bc = betweenness_centrality(g);
    ASSERT_EQ(bc.betweenness.size(), 4u);

    // Pairs:
    // (0, 2): 1
    // (0, 3): 1, 2
    // (1, 3): 2
    // CB(0)=0, CB(1)=2, CB(2)=2, CB(3)=0
    EXPECT_NEAR(bc.betweenness[0], 0.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[1], 2.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[2], 2.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[3], 0.0, 1e-9);
}

TEST(CentralityTest, BrandesBetweennessStarGraph) {
    // Star graph K_{1, 4}: center 0 connected to 1, 2, 3, 4
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {0, 2}, {0, 3}, {0, 4}};
    auto g = CSRGraph<void>::build_from_edges(5, edges, /*directed=*/false);

    auto bc = betweenness_centrality(g);
    ASSERT_EQ(bc.betweenness.size(), 5u);

    // Center node 0 lies on all 4*3/2 = 6 pairs of leaves
    EXPECT_NEAR(bc.betweenness[0], 6.0, 1e-9);
    for (std::size_t i = 1; i <= 4; ++i) {
        EXPECT_NEAR(bc.betweenness[i], 0.0, 1e-9);
    }

    // Degree centrality: center has 4 / 4 = 1.0, leaves have 1 / 4 = 0.25
    EXPECT_NEAR(bc.degree_centrality[0], 1.0, 1e-9);
    for (std::size_t i = 1; i <= 4; ++i) {
        EXPECT_NEAR(bc.degree_centrality[i], 0.25, 1e-9);
    }
}

TEST(CentralityTest, BrandesBetweennessDirectedDiamond) {
    // Directed graph:
    // 0 -> 1 -> 3
    // 0 -> 2 -> 3
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    auto g = CSRGraph<void>::build_from_edges(4, edges, /*directed=*/true);

    auto bc = betweenness_centrality(g);
    ASSERT_EQ(bc.betweenness.size(), 4u);

    // Betweenness: between 0 and 3, two shortest paths (0->1->3 and 0->2->3)
    // sigma_{03} = 2
    // sigma_{03}(1) = 1 -> delta_{03}(1) = 1/2 = 0.5
    // sigma_{03}(2) = 1 -> delta_{03}(2) = 1/2 = 0.5
    EXPECT_NEAR(bc.betweenness[0], 0.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[1], 0.5, 1e-9);
    EXPECT_NEAR(bc.betweenness[2], 0.5, 1e-9);
    EXPECT_NEAR(bc.betweenness[3], 0.0, 1e-9);
}

TEST(CentralityTest, BrandesBetweennessWeightedGraph) {
    // 0 -(1.0)-> 1 -(1.0)-> 2
    // 0 -(4.0)-> 2
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {
        {0, 1, 1.0}, {1, 2, 1.0}, {0, 2, 4.0}};
    auto g = CSRGraph<double>::build_from_weighted_edges(3, edges, /*directed=*/true);

    auto bc = betweenness_centrality(g);
    ASSERT_EQ(bc.betweenness.size(), 3u);

    // Shortest path from 0 to 2 is 0 -> 1 -> 2 (weight 2.0 < 4.0)
    // Node 1 is an intermediate on the unique shortest path
    EXPECT_NEAR(bc.betweenness[0], 0.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[1], 1.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[2], 0.0, 1e-9);
}

TEST(CentralityTest, BrandesBetweennessKrackhardtKiteGraph) {
    // 10 nodes: 0..9 (David Krackhardt 1990)
    // Canonical benchmark in igraph and NetworkX test suites
    std::vector<std::pair<VertexId, VertexId>> edges = {
        {0, 1}, {0, 2}, {0, 3}, {0, 5}, {1, 3}, {1, 4}, {1, 6}, {2, 3}, {2, 5},
        {3, 4}, {3, 5}, {3, 6}, {4, 6}, {5, 6}, {5, 7}, {6, 7}, {7, 8}, {8, 9}};
    auto g = CSRGraph<void>::build_from_edges(10, edges, /*directed=*/false);
    auto bc = betweenness_centrality(g);

    ASSERT_EQ(bc.betweenness.size(), 10u);

    // Analytical igraph unnormalized betweenness values:
    // Heather (node 7) connects {0..6} (7 nodes) to {8, 9} (2 nodes) -> 7 * 2 = 14.0
    // Ike (node 8) connects {0..7} (8 nodes) to {9} (1 node) -> 8 * 1 = 8.0
    // Jane (node 9) is a leaf -> 0.0
    EXPECT_NEAR(bc.betweenness[7], 14.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[8], 8.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[9], 0.0, 1e-9);

    // Degree centrality of Jane (deg 1) -> 1/9
    EXPECT_NEAR(bc.degree_centrality[9], 1.0 / 9.0, 1e-9);
}

TEST(CentralityTest, BrandesBetweennessBullGraph) {
    // Bull graph: triangle 1-2-3 with horns 0-1 and 3-4
    std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 2}, {2, 3}, {1, 3}, {3, 4}};
    auto g = CSRGraph<void>::build_from_edges(5, edges, /*directed=*/false);
    auto bc = betweenness_centrality(g);

    ASSERT_EQ(bc.betweenness.size(), 5u);

    // Horns 0 and 4 have betweenness 0.0
    // Node 2 is not on any shortest path between other vertices -> betweenness 0.0
    // Nodes 1 and 3 have betweenness 3.0 (for node 1: lies on (0, 2), (0, 3), (0, 4))
    EXPECT_NEAR(bc.betweenness[0], 0.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[1], 3.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[2], 0.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[3], 3.0, 1e-9);
    EXPECT_NEAR(bc.betweenness[4], 0.0, 1e-9);
}

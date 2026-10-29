#include "algoat/graph/mst.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <set>
#include <tuple>
#include <vector>

using namespace algoat::graph;

TEST(MSTTest, EmptyGraph) {
    auto g = CSRGraph<double>::build_from_weighted_edges(0, {}, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    auto serial_res = engine.compute_serial();
    EXPECT_DOUBLE_EQ(serial_res.total_weight, 0.0);
    EXPECT_TRUE(serial_res.edges.empty());
    EXPECT_EQ(serial_res.components_count, 0u);

    auto parallel_res = engine.compute_parallel(2);
    EXPECT_DOUBLE_EQ(parallel_res.total_weight, 0.0);
    EXPECT_TRUE(parallel_res.edges.empty());
    EXPECT_EQ(parallel_res.components_count, 0u);
}

TEST(MSTTest, SingleVertex) {
    auto g = CSRGraph<double>::build_from_weighted_edges(1, {}, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    auto serial_res = engine.compute_serial();
    EXPECT_DOUBLE_EQ(serial_res.total_weight, 0.0);
    EXPECT_TRUE(serial_res.edges.empty());
    EXPECT_EQ(serial_res.components_count, 1u);

    auto parallel_res = engine.compute_parallel(4);
    EXPECT_DOUBLE_EQ(parallel_res.total_weight, 0.0);
    EXPECT_TRUE(parallel_res.edges.empty());
    EXPECT_EQ(parallel_res.components_count, 1u);
}

TEST(MSTTest, SimpleTriangle) {
    // 0 - 1 (1.0), 1 - 2 (2.0), 0 - 2 (3.0)
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {
        {0, 1, 1.0}, {1, 2, 2.0}, {0, 2, 3.0}};
    auto g = CSRGraph<double>::build_from_weighted_edges(3, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    auto serial_res = engine.compute_serial();
    EXPECT_DOUBLE_EQ(serial_res.total_weight, 3.0);
    EXPECT_EQ(serial_res.edges.size(), 2u);
    EXPECT_EQ(serial_res.components_count, 1u);

    auto parallel_res = engine.compute_parallel(2);
    EXPECT_DOUBLE_EQ(parallel_res.total_weight, 3.0);
    EXPECT_EQ(parallel_res.edges.size(), 2u);
    EXPECT_EQ(parallel_res.components_count, 1u);
}

TEST(MSTTest, DisconnectedComponentsSpanningForest) {
    // Component 1: 0 - 1 (10.0), 1 - 2 (20.0), 0 - 2 (30.0) -> MST weight = 30.0, 2 edges
    // Component 2: 3 - 4 (5.0) -> MST weight = 5.0, 1 edge
    // Component 3: 5 (isolated) -> 0 edges
    // Total V = 6, Components = 3, Total MSF edges = 6 - 3 = 3 edges, Total weight = 35.0
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {
        {0, 1, 10.0}, {1, 2, 20.0}, {0, 2, 30.0}, {3, 4, 5.0}};
    auto g = CSRGraph<double>::build_from_weighted_edges(6, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    auto serial_res = engine.compute_serial();
    EXPECT_DOUBLE_EQ(serial_res.total_weight, 35.0);
    EXPECT_EQ(serial_res.edges.size(), 3u);
    EXPECT_EQ(serial_res.components_count, 3u);

    auto parallel_res = engine.compute_parallel(4);
    EXPECT_DOUBLE_EQ(parallel_res.total_weight, 35.0);
    EXPECT_EQ(parallel_res.edges.size(), 3u);
    EXPECT_EQ(parallel_res.components_count, 3u);
}

TEST(MSTTest, NegativeAndZeroWeights) {
    // 0 - 1 (-5.0), 1 - 2 (0.0), 2 - 3 (10.0), 0 - 3 (2.0)
    // MST should choose (-5.0), (0.0), (2.0) -> total weight = -3.0
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {
        {0, 1, -5.0}, {1, 2, 0.0}, {2, 3, 10.0}, {0, 3, 2.0}};
    auto g = CSRGraph<double>::build_from_weighted_edges(4, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    auto serial_res = engine.compute_serial();
    EXPECT_DOUBLE_EQ(serial_res.total_weight, -3.0);
    EXPECT_EQ(serial_res.edges.size(), 3u);

    auto parallel_res = engine.compute_parallel(2);
    EXPECT_DOUBLE_EQ(parallel_res.total_weight, -3.0);
    EXPECT_EQ(parallel_res.edges.size(), 3u);
}

TEST(MSTTest, IdenticalEdgeWeightsTieBreaking) {
    // K_4 with all weights equal to 1.0.
    // Must produce a spanning tree with exactly 3 edges and total weight 3.0 without cycles.
    std::vector<std::tuple<VertexId, VertexId, double>> edges;
    for (VertexId u = 0; u < 4; ++u) {
        for (VertexId v = u + 1; v < 4; ++v) {
            edges.emplace_back(u, v, 1.0);
        }
    }
    auto g = CSRGraph<double>::build_from_weighted_edges(4, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    auto serial_res = engine.compute_serial();
    EXPECT_DOUBLE_EQ(serial_res.total_weight, 3.0);
    EXPECT_EQ(serial_res.edges.size(), 3u);
    EXPECT_EQ(serial_res.components_count, 1u);

    auto parallel_res = engine.compute_parallel(4);
    EXPECT_DOUBLE_EQ(parallel_res.total_weight, 3.0);
    EXPECT_EQ(parallel_res.edges.size(), 3u);
    EXPECT_EQ(parallel_res.components_count, 1u);
}

TEST(MSTTest, SelfLoopsAndDuplicateEdges) {
    // Graph with self-loops: (0, 0, 100.0)
    // and multiple edges: (0, 1, 10.0), (0, 1, 2.0)
    // Expected MST chooses the lightest parallel edge (2.0) and ignores self-loop.
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {
        {0, 0, 100.0}, {0, 1, 10.0}, {0, 1, 2.0}, {1, 2, 5.0}};
    auto g = CSRGraph<double>::build_from_weighted_edges(3, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    auto serial_res = engine.compute_serial();
    EXPECT_DOUBLE_EQ(serial_res.total_weight, 7.0);
    EXPECT_EQ(serial_res.edges.size(), 2u);

    auto parallel_res = engine.compute_parallel(2);
    EXPECT_DOUBLE_EQ(parallel_res.total_weight, 7.0);
    EXPECT_EQ(parallel_res.edges.size(), 2u);
}

TEST(MSTTest, RandomGraphExactEquivalence) {
    constexpr std::size_t NUM_V = 200;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> weight_dist(1.0, 100.0);
    std::uniform_int_distribution<VertexId> v_dist(0, NUM_V - 1);

    std::vector<std::tuple<VertexId, VertexId, double>> edges;
    // Ensure connectivity by forming a random spanning tree first
    for (VertexId i = 1; i < NUM_V; ++i) {
        std::uniform_int_distribution<VertexId> prev_dist(0, i - 1);
        edges.emplace_back(prev_dist(rng), i, weight_dist(rng));
    }
    // Add additional random cross edges
    for (std::size_t i = 0; i < 2000; ++i) {
        VertexId u = v_dist(rng);
        VertexId v = v_dist(rng);
        if (u != v) {
            edges.emplace_back(u, v, weight_dist(rng));
        }
    }

    auto g = CSRGraph<double>::build_from_weighted_edges(NUM_V, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    auto serial_res = engine.compute_serial();
    EXPECT_EQ(serial_res.edges.size(), NUM_V - 1);
    EXPECT_EQ(serial_res.components_count, 1u);

    for (std::size_t num_threads : {1, 2, 4, 8}) {
        auto parallel_res = engine.compute_parallel(num_threads);
        EXPECT_NEAR(parallel_res.total_weight, serial_res.total_weight, 1e-9);
        EXPECT_EQ(parallel_res.edges.size(), NUM_V - 1);
        EXPECT_EQ(parallel_res.components_count, 1u);
    }
}

TEST(MSTTest, NaNWeightThrowsInvalidArgument) {
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {
        {0, 1, std::numeric_limits<double>::quiet_NaN()}};
    auto g = CSRGraph<double>::build_from_weighted_edges(2, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    EXPECT_THROW(static_cast<void>(engine.compute_serial()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(engine.compute_parallel(2)), std::invalid_argument);
}

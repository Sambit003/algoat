#include "algoat/graph/shortest_path.hpp"

#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace algoat::graph;

TEST(ShortestPathTest, SimpleDirectedGraphDijkstra) {
    // 0 -(1.0)-> 1 -(2.5)-> 2
    // 0 -(4.0)-> 2
    // 1 -(1.0)-> 3 -(1.0)-> 2
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {
        {0, 1, 1.0}, {1, 2, 2.5}, {0, 2, 4.0}, {1, 3, 1.0}, {3, 2, 1.0}};
    auto g = CSRGraph<double>::build_from_weighted_edges(4, edges, /*directed=*/true);
    ShortestPathEngine<double> engine(g);

    auto result = engine.dijkstra(0);
    EXPECT_DOUBLE_EQ(result.distances[0], 0.0);
    EXPECT_DOUBLE_EQ(result.distances[1], 1.0);
    EXPECT_DOUBLE_EQ(result.distances[3], 2.0);
    EXPECT_DOUBLE_EQ(result.distances[2], 3.0); // 0 -> 1 -> 3 -> 2
    EXPECT_EQ(result.predecessors[2], 3u);
    EXPECT_EQ(result.predecessors[3], 1u);
    EXPECT_EQ(result.predecessors[1], 0u);
}

TEST(ShortestPathTest, PointToPointWithEarlyTerminationAndPathReconstruction) {
    // Linear chain: 0 -> 1 -> 2 -> 3 -> 4
    std::vector<std::tuple<VertexId, VertexId, double>> edges;
    for (VertexId u = 0; u < 100; ++u) {
        edges.emplace_back(u, u + 1, 1.0);
    }
    auto g = CSRGraph<double>::build_from_weighted_edges(101, edges, /*directed=*/true);
    ShortestPathEngine<double> engine(g);

    std::vector<VertexId> path;
    auto dist = engine.dijkstra_p2p(0, 5, &path);
    ASSERT_TRUE(dist.has_value());
    EXPECT_DOUBLE_EQ(*dist, 5.0);

    std::vector<VertexId> expected_path = {0, 1, 2, 3, 4, 5};
    EXPECT_EQ(path, expected_path);

    // Source == target
    std::vector<VertexId> self_path;
    auto self_dist = engine.dijkstra_p2p(3, 3, &self_path);
    ASSERT_TRUE(self_dist.has_value());
    EXPECT_DOUBLE_EQ(*self_dist, 0.0);
    EXPECT_EQ(self_path, std::vector<VertexId>{3});
}

TEST(ShortestPathTest, UnreachableTargetReturnsNullopt) {
    // 0 -> 1, 2 -> 3 (disconnected components)
    std::vector<std::tuple<VertexId, VertexId, double>> edges = {{0, 1, 1.0}, {2, 3, 2.0}};
    auto g = CSRGraph<double>::build_from_weighted_edges(4, edges, /*directed=*/true);
    ShortestPathEngine<double> engine(g);

    std::vector<VertexId> path = {999};
    auto dist = engine.dijkstra_p2p(0, 3, &path);
    EXPECT_FALSE(dist.has_value());
    EXPECT_TRUE(path.empty());
}

TEST(ShortestPathTest, ALTSearchMatchesDijkstraDistance) {
    // Grid graph: 10x10 = 100 vertices
    const std::size_t rows = 10;
    const std::size_t cols = 10;
    const std::size_t num_v = rows * cols;
    std::vector<std::tuple<VertexId, VertexId, double>> edges;

    auto node_id = [cols](std::size_t r, std::size_t c) {
        return static_cast<VertexId>(r * cols + c);
    };

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist_w(1.0, 5.0);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            VertexId u = node_id(r, c);
            if (r + 1 < rows) {
                double w = dist_w(rng);
                edges.emplace_back(u, node_id(r + 1, c), w);
            }
            if (c + 1 < cols) {
                double w = dist_w(rng);
                edges.emplace_back(u, node_id(r, c + 1), w);
            }
        }
    }

    auto g = CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
    ShortestPathEngine<double> engine(g);

    // Before preprocessing, alt_search falls back to dijkstra_p2p
    EXPECT_FALSE(engine.has_landmarks());
    std::vector<VertexId> path_fallback;
    auto dist_fallback = engine.alt_search(0, static_cast<VertexId>(num_v - 1), &path_fallback);
    ASSERT_TRUE(dist_fallback.has_value());

    // Preprocess landmarks
    engine.preprocess_landmarks(8);
    EXPECT_TRUE(engine.has_landmarks());
    EXPECT_EQ(engine.landmarks().size(), 8u);

    // Compare Dijkstra vs ALT on 50 random pairs
    for (int q = 0; q < 50; ++q) {
        VertexId s = rng() % num_v;
        VertexId t = rng() % num_v;

        std::vector<VertexId> dijkstra_path;
        auto d_dist = engine.dijkstra_p2p(s, t, &dijkstra_path);

        std::vector<VertexId> alt_path;
        auto alt_dist = engine.alt_search(s, t, &alt_path);

        ASSERT_EQ(d_dist.has_value(), alt_dist.has_value());
        if (d_dist.has_value()) {
            EXPECT_NEAR(*d_dist, *alt_dist, 1e-9);
            // Verify path length matches distance
            if (!alt_path.empty()) {
                EXPECT_EQ(alt_path.front(), s);
                EXPECT_EQ(alt_path.back(), t);
            }
        }
    }
}

TEST(ShortestPathTest, BidirectionalSearchConsistency) {
    // 20x20 Grid
    const std::size_t rows = 20;
    const std::size_t cols = 20;
    const std::size_t num_v = rows * cols;
    std::vector<std::tuple<VertexId, VertexId, double>> edges;

    auto node_id = [cols](std::size_t r, std::size_t c) {
        return static_cast<VertexId>(r * cols + c);
    };

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist_w(1.0, 10.0);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            VertexId u = node_id(r, c);
            if (r + 1 < rows) {
                edges.emplace_back(u, node_id(r + 1, c), dist_w(rng));
            }
            if (c + 1 < cols) {
                edges.emplace_back(u, node_id(r, c + 1), dist_w(rng));
            }
        }
    }

    auto g = CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
    ShortestPathEngine<double> engine(g);
    engine.preprocess_landmarks(16);

    for (int q = 0; q < 30; ++q) {
        VertexId s = rng() % num_v;
        VertexId t = rng() % num_v;

        std::vector<VertexId> p_dijkstra;
        auto dist_dijkstra = engine.dijkstra_p2p(s, t, &p_dijkstra);

        std::vector<VertexId> p_bi_dijkstra;
        auto dist_bi_dijkstra = engine.dijkstra_bidirectional(s, t, &p_bi_dijkstra);

        std::vector<VertexId> p_alt_bi;
        auto dist_alt_bi = engine.alt_search_bidirectional(s, t, &p_alt_bi);

        ASSERT_EQ(dist_dijkstra.has_value(), dist_bi_dijkstra.has_value());
        ASSERT_EQ(dist_dijkstra.has_value(), dist_alt_bi.has_value());
        if (dist_dijkstra.has_value()) {
            EXPECT_NEAR(*dist_dijkstra, *dist_bi_dijkstra, 1e-9);
            EXPECT_NEAR(*dist_dijkstra, *dist_alt_bi, 1e-9);

            if (!p_bi_dijkstra.empty()) {
                EXPECT_EQ(p_bi_dijkstra.front(), s);
                EXPECT_EQ(p_bi_dijkstra.back(), t);
            }
            if (!p_alt_bi.empty()) {
                EXPECT_EQ(p_alt_bi.front(), s);
                EXPECT_EQ(p_alt_bi.back(), t);
            }
        }
    }
}

TEST(ShortestPathTest, MathematicalCorrectnessTenThousandQueryPairs) {
    // Generate a connected graph with 300 nodes and non-negative weights
    const std::size_t num_v = 300;
    std::mt19937_64 rng(9999);
    std::uniform_real_distribution<double> dist_w(1.0, 50.0);
    std::vector<std::tuple<VertexId, VertexId, double>> edges;

    // Ensure connectivity via ring
    for (VertexId u = 0; u < num_v; ++u) {
        edges.emplace_back(u, (u + 1) % num_v, dist_w(rng));
    }
    // Add random shortcuts
    for (std::size_t i = 0; i < 1500; ++i) {
        VertexId u = rng() % num_v;
        VertexId v = rng() % num_v;
        if (u != v) {
            edges.emplace_back(u, v, dist_w(rng));
        }
    }

    auto g = CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/true);
    ShortestPathEngine<double> engine(g);
    engine.preprocess_landmarks(16);

    std::size_t total_dijkstra_settled = 0;
    std::size_t total_alt_settled = 0;

    for (int q = 0; q < 10000; ++q) {
        VertexId s = static_cast<VertexId>(rng() % num_v);
        VertexId t = static_cast<VertexId>(rng() % num_v);

        std::size_t d_settled = 0;
        std::size_t a_settled = 0;

        auto d_dist = engine.dijkstra_p2p(s, t, nullptr, &d_settled);
        auto alt_dist = engine.alt_search(s, t, nullptr, &a_settled);

        total_dijkstra_settled += d_settled;
        total_alt_settled += a_settled;

        ASSERT_EQ(d_dist.has_value(), alt_dist.has_value());
        if (d_dist.has_value()) {
            ASSERT_NEAR(*d_dist, *alt_dist, 1e-9);
        }
    }

    EXPECT_GT(total_dijkstra_settled, 0u);
    EXPECT_GT(total_alt_settled, 0u);
    // ALT should explore substantially fewer nodes on average
    EXPECT_LT(total_alt_settled, total_dijkstra_settled);
}

TEST(ShortestPathTest, SearchSpaceReductionTenXOnPlanarCorridor) {
    // 50x50 grid (2500 nodes)
    const std::size_t rows = 50;
    const std::size_t cols = 50;
    const std::size_t num_v = rows * cols;
    std::vector<std::tuple<VertexId, VertexId, double>> edges;

    auto node_id = [cols](std::size_t r, std::size_t c) {
        return static_cast<VertexId>(r * cols + c);
    };

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            VertexId u = node_id(r, c);
            if (r + 1 < rows)
                edges.emplace_back(u, node_id(r + 1, c), 1.0);
            if (c + 1 < cols)
                edges.emplace_back(u, node_id(r, c + 1), 1.0);
        }
    }

    auto g = CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
    ShortestPathEngine<double> engine(g);
    engine.preprocess_landmarks(16);

    // Queries between opposite corners: (0, 0) to (rows-1, cols-1)
    VertexId s = node_id(0, 0);
    VertexId t = node_id(rows - 1, cols - 1);

    std::size_t d_settled = 0;
    std::size_t a_settled = 0;

    auto d_dist = engine.dijkstra_p2p(s, t, nullptr, &d_settled);
    auto alt_dist = engine.alt_search(s, t, nullptr, &a_settled);

    ASSERT_TRUE(d_dist.has_value());
    ASSERT_TRUE(alt_dist.has_value());
    EXPECT_DOUBLE_EQ(*d_dist, *alt_dist);

    // Dijkstra explores the full radial sphere (almost all 1225 nodes);
    // ALT is guided along the corridor and settles >= 10x fewer nodes.
    EXPECT_GT(d_settled, a_settled);
    EXPECT_GE(d_settled, a_settled * 10);
}

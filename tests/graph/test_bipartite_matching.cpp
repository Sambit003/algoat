#include "algoat/graph/bipartite_matching.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <unordered_set>
#include <vector>

using namespace algoat::graph;

namespace {

// Helper to verify König's Theorem and Independent Set invariants on a graph and matching result
void verify_matching_and_dualities(const BipartiteGraph& graph,
                                   const BipartiteMatchingResult& matching) {
    const std::size_t num_left = graph.num_left();
    const std::size_t num_right = graph.num_right();

    EXPECT_EQ(matching.match_left.size(), num_left);
    EXPECT_EQ(matching.match_right.size(), num_right);

    // 1. Verify matching validity: edges exist and reciprocal
    std::size_t count_left = 0;
    for (VertexId u = 0; u < num_left; ++u) {
        const VertexId v = matching.match_left[u];
        if (v != BipartiteGraph::kUnmatched) {
            EXPECT_LT(v, num_right);
            EXPECT_EQ(matching.match_right[v], u);
            // Verify edge exists in adjacency list via adj() span
            const auto nbrs = graph.adj(u);
            EXPECT_NE(std::find(nbrs.begin(), nbrs.end(), v), nbrs.end());
            count_left++;
        }
    }
    EXPECT_EQ(count_left, matching.matching_size);

    std::size_t count_right = 0;
    for (VertexId v = 0; v < num_right; ++v) {
        const VertexId u = matching.match_right[v];
        if (u != BipartiteGraph::kUnmatched) {
            EXPECT_LT(u, num_left);
            EXPECT_EQ(matching.match_left[u], v);
            count_right++;
        }
    }
    EXPECT_EQ(count_right, matching.matching_size);

    // 2. Minimum Vertex Cover (König's Theorem)
    const auto [cover_l, cover_r] = graph.minimum_vertex_cover(matching);

    // Cardinality equality: |Cover| == |Matching|
    EXPECT_EQ(cover_l.size() + cover_r.size(), matching.matching_size);

    std::vector<bool> in_cover_l(num_left, false);
    for (VertexId u : cover_l) {
        EXPECT_LT(u, num_left);
        in_cover_l[u] = true;
    }
    std::vector<bool> in_cover_r(num_right, false);
    for (VertexId v : cover_r) {
        EXPECT_LT(v, num_right);
        in_cover_r[v] = true;
    }

    // Every edge must be covered: (u in Cover_L) or (v in Cover_R)
    for (VertexId u = 0; u < num_left; ++u) {
        for (VertexId v : graph.adj(u)) {
            EXPECT_TRUE(in_cover_l[u] || in_cover_r[v])
                << "Edge (" << u << ", " << v << ") is uncovered by the vertex cover";
        }
    }

    // 3. Maximum Independent Set (Complementation)
    const auto [indep_l, indep_r] = graph.maximum_independent_set(matching);

    // Gallai's Identity: |Indep| == |V| - |Cover| == |L| + |R| - |Matching|
    EXPECT_EQ(indep_l.size() + indep_r.size(), num_left + num_right - matching.matching_size);

    std::vector<bool> in_indep_r(num_right, false);
    for (VertexId v : indep_r) {
        EXPECT_LT(v, num_right);
        in_indep_r[v] = true;
    }

    // No edge can connect two vertices in the independent set
    for (VertexId u : indep_l) {
        EXPECT_LT(u, num_left);
        for (VertexId v : graph.adj(u)) {
            EXPECT_FALSE(in_indep_r[v])
                << "Edge (" << u << ", " << v << ") connects two independent set vertices";
        }
    }

    // 4. Berge's Lemma invariant: no augmenting paths exist
    EXPECT_TRUE(BipartiteMatchingTestAccess::verify_berge_invariant(graph, matching));
}

} // namespace

TEST(BipartiteMatchingTest, EmptyGraph) {
    BipartiteGraph g(0, 0);
    EXPECT_EQ(g.num_left(), 0);
    EXPECT_EQ(g.num_right(), 0);
    EXPECT_EQ(g.num_edges(), 0);

    const auto res = g.maximum_matching();
    EXPECT_EQ(res.matching_size, 0);
    EXPECT_TRUE(res.match_left.empty());
    EXPECT_TRUE(res.match_right.empty());

    const auto [cov_l, cov_r] = g.minimum_vertex_cover(res);
    EXPECT_TRUE(cov_l.empty());
    EXPECT_TRUE(cov_r.empty());

    const auto [ind_l, ind_r] = g.maximum_independent_set(res);
    EXPECT_TRUE(ind_l.empty());
    EXPECT_TRUE(ind_r.empty());

    EXPECT_TRUE(BipartiteMatchingTestAccess::verify_berge_invariant(g, res));
}

TEST(BipartiteMatchingTest, EdgesWithoutVerticesThrows) {
    BipartiteGraph g(2, 3);
    EXPECT_THROW(g.add_edge(2, 0), std::out_of_range);
    EXPECT_THROW(g.add_edge(0, 3), std::out_of_range);
    EXPECT_NO_THROW(g.add_edge(1, 2));
    EXPECT_EQ(g.num_edges(), 1);
}

TEST(BipartiteMatchingTest, DisconnectedVerticesNoEdges) {
    BipartiteGraph g(4, 5);
    const auto res = g.maximum_matching();
    EXPECT_EQ(res.matching_size, 0);
    verify_matching_and_dualities(g, res);
}

TEST(BipartiteMatchingTest, SingleEdge) {
    BipartiteGraph g(1, 1);
    g.add_edge(0, 0);
    const auto res = g.maximum_matching();
    EXPECT_EQ(res.matching_size, 1);
    EXPECT_EQ(res.match_left[0], 0);
    EXPECT_EQ(res.match_right[0], 0);
    verify_matching_and_dualities(g, res);
}

TEST(BipartiteMatchingTest, DisjointEdges) {
    constexpr std::size_t n = 5;
    BipartiteGraph g(n, n);
    for (VertexId i = 0; i < n; ++i) {
        g.add_edge(i, i);
    }
    const auto res = g.maximum_matching();
    EXPECT_EQ(res.matching_size, n);
    verify_matching_and_dualities(g, res);
}

TEST(BipartiteMatchingTest, StarGraph) {
    // One left vertex connected to 4 right vertices
    BipartiteGraph g(1, 4);
    for (VertexId v = 0; v < 4; ++v) {
        g.add_edge(0, v);
    }
    const auto res = g.maximum_matching();
    EXPECT_EQ(res.matching_size, 1);
    EXPECT_EQ(res.match_left[0], 0);
    verify_matching_and_dualities(g, res);
}

TEST(BipartiteMatchingTest, AlternatingPathGraph) {
    // Path: L0 - R0 - L1 - R1 - L2
    BipartiteGraph g(3, 2);
    g.add_edge(0, 0);
    g.add_edge(1, 0);
    g.add_edge(1, 1);
    g.add_edge(2, 1);

    const auto res = g.maximum_matching();
    EXPECT_EQ(res.matching_size, 2);
    verify_matching_and_dualities(g, res);
}

TEST(BipartiteMatchingTest, CompleteBipartiteGraph) {
    constexpr std::size_t nl = 4;
    constexpr std::size_t nr = 6;
    BipartiteGraph g(nl, nr);
    for (VertexId u = 0; u < nl; ++u) {
        for (VertexId v = 0; v < nr; ++v) {
            g.add_edge(u, v);
        }
    }
    const auto res = g.maximum_matching();
    EXPECT_EQ(res.matching_size, std::min(nl, nr));
    verify_matching_and_dualities(g, res);
}

TEST(BipartiteMatchingTest, BergeInvariantDetectsSuboptimalMatching) {
    // Triangle-free 4-cycle path: L0 - R0, L0 - R1, L1 - R0, L1 - R1
    BipartiteGraph g(2, 2);
    g.add_edge(0, 0);
    g.add_edge(0, 1);
    g.add_edge(1, 0);
    g.add_edge(1, 1);

    const auto optimal = g.maximum_matching();
    EXPECT_EQ(optimal.matching_size, 2);
    EXPECT_TRUE(BipartiteMatchingTestAccess::verify_berge_invariant(g, optimal));

    // Construct suboptimal matching of size 1: match (0, 0) only
    BipartiteMatchingResult suboptimal;
    suboptimal.matching_size = 1;
    suboptimal.match_left = {0, BipartiteGraph::kUnmatched};
    suboptimal.match_right = {0, BipartiteGraph::kUnmatched};

    // Berge invariant must report FALSE because augmenting path L1 -> R1 exists!
    EXPECT_FALSE(BipartiteMatchingTestAccess::verify_berge_invariant(g, suboptimal));
}

TEST(BipartiteMatchingTest, TenThousandRandomGraphsKonigDuality) {
    std::mt19937_64 rng(42);

    for (int trial = 0; trial < 10000; ++trial) {
        const std::size_t nl = (rng() % 16) + 1;
        const std::size_t nr = (rng() % 16) + 1;
        BipartiteGraph g(nl, nr);

        const double edge_prob = static_cast<double>(rng() % 100) / 100.0;
        for (VertexId u = 0; u < nl; ++u) {
            for (VertexId v = 0; v < nr; ++v) {
                if (std::generate_canonical<double, 10>(rng) < edge_prob) {
                    g.add_edge(u, v);
                }
            }
        }

        const auto res = g.maximum_matching();
        verify_matching_and_dualities(g, res);
    }
}

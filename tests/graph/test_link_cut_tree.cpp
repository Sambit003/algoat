#include "../core/pmr_test_helpers.hpp"
#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/disjoint_set.hpp"
#include "algoat/graph/link_cut_tree.hpp"
#include "algoat/graph/mst.hpp"
#include "link_cut_tree_test_helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

ALGOAT_INSTALL_GLOBAL_ALLOC_COUNTER()

using algoat::graph::CSRGraph;
using algoat::graph::DisjointSetUnion;
using algoat::graph::LinkCutTree;
using algoat::graph::MinimumSpanningTreeEngine;
using algoat::graph::VertexId;
using algoat::graph::testing::DynamicMstMaintainer;
using algoat::graph::testing::ForestOracle;

TEST(LinkCutTreeTest, EmptyAndSingletonConstruction) {
    LinkCutTree<> empty(0);
    EXPECT_EQ(empty.size(), 0U);

    LinkCutTree<> single(1);
    EXPECT_EQ(single.size(), 1U);
    EXPECT_EQ(single.find_root(0), 0U);
    EXPECT_TRUE(single.connected(0, 0));

    LinkCutTree<> forest(5);
    EXPECT_EQ(forest.size(), 5U);
}

TEST(LinkCutTreeTest, OutOfBoundsThrows) {
    LinkCutTree<> empty(0);
    EXPECT_THROW(static_cast<void>(empty.find_root(0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(empty.connected(0, 0)), std::out_of_range);
    EXPECT_THROW(empty.link(0, 0), std::out_of_range);
    EXPECT_THROW(empty.cut(0, 0), std::out_of_range);
    EXPECT_THROW(empty.make_root(0), std::out_of_range);
    EXPECT_THROW(empty.update_vertex(0, 10), std::out_of_range);
    EXPECT_THROW(static_cast<void>(empty.query_path_sum(0, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(empty.query_path_min(0, 0)), std::out_of_range);

    LinkCutTree<> lct(4);
    EXPECT_THROW(static_cast<void>(lct.find_root(4)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(lct.connected(0, 4)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(lct.connected(4, 0)), std::out_of_range);
    EXPECT_THROW(lct.link(4, 1), std::out_of_range);
    EXPECT_THROW(lct.link(1, 4), std::out_of_range);
    EXPECT_THROW(lct.cut(4, 1), std::out_of_range);
    EXPECT_THROW(lct.cut(1, 4), std::out_of_range);
    EXPECT_THROW(lct.make_root(4), std::out_of_range);
    EXPECT_THROW(lct.update_vertex(4, 5), std::out_of_range);
    EXPECT_THROW(static_cast<void>(lct.query_path_sum(4, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(lct.query_path_min(0, 4)), std::out_of_range);
}

TEST(LinkCutTreeTest, MaxSizeThrowsLengthError) {
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(LinkCutTree<> too_large(kMax), std::length_error);
}

TEST(LinkCutTreeTest, SingletonConnectivityAndRoots) {
    LinkCutTree<> lct(4);
    for (std::size_t u = 0; u < 4; ++u) {
        EXPECT_EQ(lct.find_root(u), u);
        for (std::size_t v = 0; v < 4; ++v) {
            EXPECT_EQ(lct.connected(u, v), u == v);
        }
    }
}

TEST(LinkCutTreeTest, LinkBuildsChainAndTreeWithFindRootPreserved) {
    LinkCutTree<> lct(6);
    // Tree A rooted at 0: 0 <- 1 <- 2, 1 <- 3
    lct.link(1, 0);
    lct.link(2, 1);
    lct.link(3, 1);
    // Tree B rooted at 4: 4 <- 5
    lct.link(5, 4);

    for (std::size_t u : {0U, 1U, 2U, 3U}) {
        EXPECT_EQ(lct.find_root(u), 0U);
        EXPECT_TRUE(lct.connected(u, 0));
        EXPECT_TRUE(lct.connected(0, u));
        EXPECT_FALSE(lct.connected(u, 4));
        EXPECT_FALSE(lct.connected(u, 5));
    }
    EXPECT_TRUE(lct.connected(2, 3));
    EXPECT_EQ(lct.find_root(4), 4U);
    EXPECT_EQ(lct.find_root(5), 4U);
    EXPECT_TRUE(lct.connected(4, 5));

    // Re-checking find_root and connected must preserve roots
    EXPECT_EQ(lct.find_root(2), 0U);
    EXPECT_EQ(lct.find_root(3), 0U);
    EXPECT_EQ(lct.find_root(5), 4U);
}

TEST(LinkCutTreeTest, LinkAttachesMultiNodeRootSubtree) {
    LinkCutTree<> lct(5);
    lct.link(1, 0);
    lct.link(2, 1);
    lct.link(4, 3);

    // 0 is root of {0,1,2}; attach it under 4 (in tree rooted at 3)
    lct.link(0, 4);
    for (std::size_t u = 0; u < 5; ++u) {
        EXPECT_EQ(lct.find_root(u), 3U);
        EXPECT_TRUE(lct.connected(2, u));
    }
}

TEST(LinkCutTreeTest, InvalidLinkThrowsWithoutMutatingState) {
    LinkCutTree<> lct(5);
    lct.link(1, 0);
    lct.link(2, 1);
    lct.link(4, 3);

    // Self-link
    EXPECT_THROW(lct.link(0, 0), std::invalid_argument);
    EXPECT_THROW(lct.link(2, 2), std::invalid_argument);

    // Same-tree cycle link (root to descendant)
    EXPECT_THROW(lct.link(0, 2), std::invalid_argument);

    // Non-root u linked to another tree (strict precondition from spec)
    EXPECT_THROW(lct.link(1, 3), std::invalid_argument);
    EXPECT_THROW(lct.link(2, 4), std::invalid_argument);

    // Forest topology and roots remain completely unchanged
    EXPECT_EQ(lct.find_root(0), 0U);
    EXPECT_EQ(lct.find_root(1), 0U);
    EXPECT_EQ(lct.find_root(2), 0U);
    EXPECT_EQ(lct.find_root(3), 3U);
    EXPECT_EQ(lct.find_root(4), 3U);
    EXPECT_FALSE(lct.connected(1, 3));
    EXPECT_FALSE(lct.connected(2, 4));
}

TEST(LinkCutTreeTest, CutSplitsTreeInBothOrdersAndPreservesUpperRoot) {
    LinkCutTree<> lct(5);
    // 0 <- 1 <- 2 <- 3, and 1 <- 4
    lct.link(1, 0);
    lct.link(2, 1);
    lct.link(3, 2);
    lct.link(4, 1);

    // Cut parent-then-child order: edge (1, 2)
    lct.cut(1, 2);
    EXPECT_EQ(lct.find_root(0), 0U);
    EXPECT_EQ(lct.find_root(1), 0U);
    EXPECT_EQ(lct.find_root(4), 0U);
    EXPECT_EQ(lct.find_root(2), 2U);
    EXPECT_EQ(lct.find_root(3), 2U);
    EXPECT_FALSE(lct.connected(1, 2));
    EXPECT_TRUE(lct.connected(2, 3));

    // Cut child-then-parent order: edge (3, 2)
    lct.cut(3, 2);
    EXPECT_EQ(lct.find_root(2), 2U);
    EXPECT_EQ(lct.find_root(3), 3U);
    EXPECT_FALSE(lct.connected(2, 3));

    // Detached subtree root 2 can be re-linked elsewhere
    lct.link(2, 4);
    EXPECT_EQ(lct.find_root(2), 0U);
    EXPECT_TRUE(lct.connected(0, 2));
}

TEST(LinkCutTreeTest, InvalidCutThrowsWithoutMutatingState) {
    LinkCutTree<> lct(5);
    lct.link(1, 0);
    lct.link(2, 1);
    lct.link(4, 3);

    EXPECT_THROW(lct.cut(1, 1), std::invalid_argument);
    EXPECT_THROW(lct.cut(0, 2), std::invalid_argument);
    EXPECT_THROW(lct.cut(2, 0), std::invalid_argument);
    EXPECT_THROW(lct.cut(0, 4), std::invalid_argument);

    EXPECT_EQ(lct.find_root(0), 0U);
    EXPECT_EQ(lct.find_root(1), 0U);
    EXPECT_EQ(lct.find_root(2), 0U);
    EXPECT_EQ(lct.find_root(3), 3U);
    EXPECT_EQ(lct.find_root(4), 3U);
    EXPECT_TRUE(lct.connected(0, 2));
    EXPECT_FALSE(lct.connected(0, 4));
}

TEST(LinkCutTreeTest, MakeRootEvertsTreeAndSupportsSubsequentLinkAndCut) {
    LinkCutTree<> lct(5);
    lct.link(1, 0);
    lct.link(2, 1);
    lct.link(3, 1);

    lct.make_root(2);
    for (std::size_t u : {0U, 1U, 2U, 3U}) {
        EXPECT_EQ(lct.find_root(u), 2U);
    }

    // Idempotent make_root on current root
    lct.make_root(2);
    EXPECT_EQ(lct.find_root(0), 2U);

    // Reroot to 3, then link singleton 4 under 2, then cut (1, 3)
    lct.make_root(3);
    EXPECT_EQ(lct.find_root(0), 3U);
    EXPECT_THROW(lct.link(0, 4), std::invalid_argument); // 0 is no longer root

    lct.link(4, 2);
    EXPECT_EQ(lct.find_root(4), 3U);

    lct.cut(1, 3);
    EXPECT_EQ(lct.find_root(3), 3U);
    EXPECT_EQ(lct.find_root(0), 1U);
    EXPECT_EQ(lct.find_root(1), 1U);
    EXPECT_EQ(lct.find_root(2), 1U);
    EXPECT_EQ(lct.find_root(4), 1U);
    EXPECT_FALSE(lct.connected(3, 1));
    EXPECT_TRUE(lct.connected(0, 4));
}

TEST(LinkCutTreeTest, PathSumAndMinAcrossTreeAndPreservesRoots) {
    LinkCutTree<> lct(7);
    // Values: 0:10, 1:-3, 2:7, 3:2, 4:-8, 5:100, 6:4
    const int64_t vals[7] = {10, -3, 7, 2, -8, 100, 4};
    for (std::size_t i = 0; i < 7; ++i) {
        lct.update_vertex(i, vals[i]);
    }
    // Tree rooted at 0: 0 <- 1 <- 2 <- 4; 1 <- 3; 0 <- 5 <- 6
    lct.link(1, 0);
    lct.link(2, 1);
    lct.link(4, 2);
    lct.link(3, 1);
    lct.link(5, 0);
    lct.link(6, 5);

    // Single vertex
    EXPECT_EQ(lct.query_path_sum(4, 4), -8);
    EXPECT_EQ(lct.query_path_min(4, 4), -8);

    // Ancestor-descendant path 0..4: {10, -3, 7, -8}
    EXPECT_EQ(lct.query_path_sum(0, 4), 6);
    EXPECT_EQ(lct.query_path_sum(4, 0), 6);
    EXPECT_EQ(lct.query_path_min(0, 4), -8);
    EXPECT_EQ(lct.query_path_min(4, 0), -8);

    // Cross-LCA path 4..3: {4:-8, 2:7, 1:-3, 3:2}
    EXPECT_EQ(lct.query_path_sum(4, 3), -2);
    EXPECT_EQ(lct.query_path_min(4, 3), -8);

    // Cross-root path 4..6: {4:-8, 2:7, 1:-3, 0:10, 5:100, 6:4}
    EXPECT_EQ(lct.query_path_sum(4, 6), 110);
    EXPECT_EQ(lct.query_path_sum(6, 4), 110);
    EXPECT_EQ(lct.query_path_min(4, 6), -8);
    EXPECT_EQ(lct.query_path_min(3, 6), -3); // {3:2, 1:-3, 0:10, 5:100, 6:4}

    // Roots must remain 0 after all queries
    for (std::size_t i = 0; i < 7; ++i) {
        EXPECT_EQ(lct.find_root(i), 0U);
    }

    // Update vertex 1 from -3 to 20, verify updated paths and root preservation
    lct.update_vertex(1, 20);
    EXPECT_EQ(lct.query_path_sum(4, 3), 21); // -8 + 7 + 20 + 2
    EXPECT_EQ(lct.query_path_min(3, 6), 2);  // {2, 20, 10, 100, 4}
    EXPECT_EQ(lct.find_root(6), 0U);

    // Reroot to 4, then query, cut, and re-link
    lct.make_root(4);
    EXPECT_EQ(lct.query_path_sum(3, 6), 136); // 2 + 20 + 10 + 100 + 4
    EXPECT_EQ(lct.query_path_min(3, 6), 2);
    EXPECT_EQ(lct.find_root(6), 4U);

    lct.cut(0, 5);
    EXPECT_EQ(lct.query_path_sum(0, 4), 29); // 10 + 20 + 7 - 8
    EXPECT_EQ(lct.query_path_sum(5, 6), 104);
    EXPECT_EQ(lct.query_path_min(5, 6), 4);
}

TEST(LinkCutTreeTest, DisconnectedPathQueryThrowsWithoutMutatingRoots) {
    LinkCutTree<> lct(5);
    lct.update_vertex(0, 10);
    lct.update_vertex(1, 20);
    lct.update_vertex(2, 30);
    lct.update_vertex(3, 40);
    lct.update_vertex(4, 50);

    lct.link(1, 0);
    lct.link(2, 1);
    lct.link(4, 3);
    lct.make_root(2);

    EXPECT_THROW(static_cast<void>(lct.query_path_sum(0, 3)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(lct.query_path_min(2, 4)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(lct.query_path_sum(4, 1)), std::invalid_argument);

    EXPECT_EQ(lct.find_root(0), 2U);
    EXPECT_EQ(lct.find_root(1), 2U);
    EXPECT_EQ(lct.find_root(3), 3U);
    EXPECT_EQ(lct.find_root(4), 3U);
}

TEST(LinkCutTreeTest, TypedValuesInt32DoubleNanAndCustomMonoid) {
    LinkCutTree<int32_t> lct32(3);
    lct32.update_vertex(0, -100);
    lct32.update_vertex(1, 45);
    lct32.update_vertex(2, -12);
    lct32.link(1, 0);
    lct32.link(2, 1);
    EXPECT_EQ(lct32.query_path_sum(0, 2), -67);
    EXPECT_EQ(lct32.query_path_min(1, 2), -12);

    LinkCutTree<double> lctd(3);
    lctd.update_vertex(0, 1.5);
    lctd.update_vertex(1, -2.25);
    lctd.update_vertex(2, 4.0);
    lctd.link(1, 0);
    lctd.link(2, 1);
    EXPECT_DOUBLE_EQ(lctd.query_path_sum(0, 2), 3.25);
    EXPECT_DOUBLE_EQ(lctd.query_path_min(0, 2), -2.25);
    EXPECT_THROW(lctd.update_vertex(1, std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);

    struct CustomMetric {
        int score{0};
        uint32_t tag{0};

        [[nodiscard]] CustomMetric operator+(const CustomMetric& rhs) const noexcept {
            return CustomMetric{score + rhs.score, tag ^ rhs.tag};
        }

        [[nodiscard]] bool operator<(const CustomMetric& rhs) const noexcept {
            return score < rhs.score;
        }
    };

    LinkCutTree<CustomMetric> custom(3);
    custom.update_vertex(0, {5, 1U});
    custom.update_vertex(1, {2, 2U});
    custom.update_vertex(2, {9, 4U});
    custom.link(1, 0);
    custom.link(2, 1);
    const auto sum = custom.query_path_sum(0, 2);
    EXPECT_EQ(sum.score, 16);
    EXPECT_EQ(sum.tag, 7U);
    const auto mn = custom.query_path_min(0, 2);
    EXPECT_EQ(mn.score, 2);
    EXPECT_EQ(mn.tag, 2U);
}

namespace {

void run_differential_workload(std::size_t n, std::size_t num_ops, uint64_t seed) {
    LinkCutTree<int64_t> lct(n);
    ForestOracle<int64_t> oracle(n);
    std::vector<std::pair<std::size_t, std::size_t>> active_edges;
    active_edges.reserve(n);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> val_dist(-100000, 100000);
    std::uniform_int_distribution<std::size_t> vertex_dist(0, n - 1);
    std::uniform_int_distribution<std::size_t> op_dist(0, 99);

    for (std::size_t i = 0; i < n; ++i) {
        const int64_t init_val = val_dist(rng);
        lct.update_vertex(i, init_val);
        oracle.update_vertex(i, init_val);
    }

    for (std::size_t step = 0; step < num_ops; ++step) {
        const std::size_t op = op_dist(rng);
        const std::size_t u = vertex_dist(rng);
        const std::size_t v = vertex_dist(rng);

        if (op < 22) {
            // Link (sometimes re-rooting u to root first, sometimes testing root requirement)
            if (u != v && !oracle.connected(u, v)) {
                if (oracle.is_root(u)) {
                    lct.link(u, v);
                    oracle.link(u, v);
                    active_edges.emplace_back(u, v);
                } else {
                    EXPECT_THROW(lct.link(u, v), std::invalid_argument);
                    lct.make_root(u);
                    oracle.make_root(u);
                    lct.link(u, v);
                    oracle.link(u, v);
                    active_edges.emplace_back(u, v);
                }
            } else if (u == v || oracle.connected(u, v)) {
                EXPECT_THROW(lct.link(u, v), std::invalid_argument);
            }
        } else if (op < 36) {
            // Cut an existing edge (60% existing, 40% arbitrary pair)
            if (!active_edges.empty() && (step % 5 != 0)) {
                const std::size_t idx = rng() % active_edges.size();
                auto [eu, ev] = active_edges[idx];
                if ((step & 1U) != 0U) {
                    std::swap(eu, ev);
                }
                lct.cut(eu, ev);
                oracle.cut(eu, ev);
                active_edges[idx] = active_edges.back();
                active_edges.pop_back();
            } else if (u != v && oracle.has_edge(u, v)) {
                lct.cut(u, v);
                oracle.cut(u, v);
                for (std::size_t idx = 0; idx < active_edges.size(); ++idx) {
                    const auto [a, b] = active_edges[idx];
                    if ((a == u && b == v) || (a == v && b == u)) {
                        active_edges[idx] = active_edges.back();
                        active_edges.pop_back();
                        break;
                    }
                }
            } else {
                EXPECT_THROW(lct.cut(u, v), std::invalid_argument);
            }
        } else if (op < 46) {
            lct.make_root(u);
            oracle.make_root(u);
        } else if (op < 56) {
            const int64_t new_val = val_dist(rng);
            lct.update_vertex(u, new_val);
            oracle.update_vertex(u, new_val);
        } else if (op < 70) {
            ASSERT_EQ(lct.connected(u, v), oracle.connected(u, v));
        } else if (op < 80) {
            ASSERT_EQ(lct.find_root(u), oracle.find_root(u));
        } else {
            if (oracle.connected(u, v)) {
                const auto [exp_sum, exp_min] = oracle.query_path(u, v);
                if (op < 90) {
                    ASSERT_EQ(lct.query_path_sum(u, v), exp_sum);
                } else {
                    ASSERT_EQ(lct.query_path_min(u, v), exp_min);
                }
            } else {
                EXPECT_THROW(static_cast<void>(lct.query_path_sum(u, v)), std::invalid_argument);
                EXPECT_THROW(static_cast<void>(lct.query_path_min(u, v)), std::invalid_argument);
            }
        }
    }

    for (std::size_t u = 0; u < n; ++u) {
        ASSERT_EQ(lct.find_root(u), oracle.find_root(u));
    }
}

} // namespace

TEST(LinkCutTreeTest, DifferentialMillionOpsMatchesBfsOracle) {
    // 1,000,000 total dynamic operations split across high-churn small forests and medium forests
    run_differential_workload(24, 500000, 0xA160A717401ULL);
    run_differential_workload(96, 500000, 0xA160A717402ULL);
}

TEST(LinkCutTreeTest, ZeroHeapAllocationsAfterConstruction) {
    constexpr std::size_t kN = 1024;
    LinkCutTree<int64_t> lct(kN);

    algoat::testing::ScopedGlobalAllocTracker tracker;

    for (std::size_t i = 1; i < kN; ++i) {
        lct.update_vertex(i, static_cast<int64_t>(i) - 500);
        lct.link(i, (i - 1) / 2);
    }

    int64_t sink = 0;
    for (std::size_t step = 0; step < 10000; ++step) {
        const std::size_t u = (step * 131 + 17) % kN;
        const std::size_t v = (step * 977 + 53) % kN;
        sink ^= static_cast<int64_t>(lct.connected(u, v));
        sink ^= static_cast<int64_t>(lct.find_root(u));
        if (u != v) {
            sink ^= lct.query_path_sum(u, v);
            sink ^= lct.query_path_min(u, v);
        }
        if (step % 7 == 0 && u > 0) {
            const std::size_t p = (u - 1) / 2;
            lct.cut(u, p);
            lct.make_root(u);
            lct.link(u, p);
        }
        lct.update_vertex(v, static_cast<int64_t>(step % 997) - 400);
    }
    (void)sink;

    EXPECT_EQ(tracker.count(), 0U);
}

TEST(LinkCutTreeTest, AdversarialLongChainFindRootAndMakeRoot) {
    constexpr std::size_t kN = 100000;
    LinkCutTree<int64_t> lct(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        lct.update_vertex(i, static_cast<int64_t>(i));
    }
    for (std::size_t i = 1; i < kN; ++i) {
        lct.link(i, i - 1);
    }

    // Repeated deepest-node find_root would degrade to O(N^2) if find_root omitted splay(root)
    for (std::size_t iter = 0; iter < 10000; ++iter) {
        ASSERT_EQ(lct.find_root(kN - 1), 0U);
    }

    // Alternating everts across 10^5-deep tree would stack-overflow recursive push_down
    for (std::size_t iter = 0; iter < 500; ++iter) {
        lct.make_root(kN - 1);
        ASSERT_EQ(lct.find_root(0), kN - 1);
        lct.make_root(0);
        ASSERT_EQ(lct.find_root(kN - 1), 0U);
    }

    const int64_t expected_sum = static_cast<int64_t>(kN - 1) * static_cast<int64_t>(kN) / 2;
    EXPECT_EQ(lct.query_path_sum(0, kN - 1), expected_sum);
    EXPECT_EQ(lct.query_path_min(0, kN - 1), 0);
}

TEST(LinkCutTreeTest, DynamicMstMaintenanceMatchesKruskal) {
    constexpr std::size_t kV = 40;
    DynamicMstMaintainer dyn_mst(kV);
    std::vector<std::tuple<VertexId, VertexId, int64_t>> all_edges;

    auto verify_against_kruskal = [&]() {
        // Direct multigraph Kruskal with DisjointSetUnion
        auto sorted = all_edges;
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
        DisjointSetUnion dsu(kV);
        int64_t kruskal_weight = 0;
        for (const auto& [eu, ev, ew] : sorted) {
            if (eu != ev && dsu.unite(eu, ev)) {
                kruskal_weight += ew;
            }
        }
        EXPECT_EQ(dyn_mst.total_weight(), kruskal_weight);
        EXPECT_EQ(dyn_mst.components_count(), dsu.num_components());

        // Also cross-check against CSRGraph + MinimumSpanningTreeEngine using min parallel weights
        std::map<std::pair<VertexId, VertexId>, int64_t> min_simple_edges;
        for (const auto& [eu, ev, ew] : all_edges) {
            if (eu == ev) {
                continue;
            }
            const auto key = std::minmax(eu, ev);
            auto [it, inserted] = min_simple_edges.emplace(key, ew);
            if (!inserted) {
                it->second = std::min(it->second, ew);
            }
        }
        std::vector<std::tuple<VertexId, VertexId, int64_t>> simple_edges;
        simple_edges.reserve(min_simple_edges.size());
        for (const auto& [uv, ew] : min_simple_edges) {
            simple_edges.emplace_back(uv.first, uv.second, ew);
        }
        auto csr =
            CSRGraph<int64_t>::build_from_weighted_edges(kV, simple_edges, /*directed=*/false);
        const auto csr_mst = MinimumSpanningTreeEngine<int64_t>(csr).compute_serial();
        EXPECT_EQ(dyn_mst.total_weight(), csr_mst.total_weight);
        EXPECT_EQ(dyn_mst.components_count(), csr_mst.components_count);
    };

    // Seed handcrafted edges: disconnected components, negative weights, tie, parallel heavier edge
    const std::vector<std::tuple<VertexId, VertexId, int64_t>> seed_edges = {
        {0, 1, 10}, {1, 2, 20},   {0, 2, 5}, {3, 4, -7},
        {0, 2, 50}, {3, 3, -100}, {4, 5, 3}, {2, 3, 4}};
    for (const auto& [u, v, w] : seed_edges) {
        dyn_mst.insert_edge(u, v, w);
        all_edges.emplace_back(u, v, w);
        verify_against_kruskal();
    }

    // Stream 300 random weighted edge insertions and verify periodically
    std::mt19937_64 rng(0xD717AC71CULL);
    std::uniform_int_distribution<std::size_t> v_dist(0, kV - 1);
    std::uniform_int_distribution<int64_t> w_dist(-500, 500);
    for (std::size_t step = 0; step < 300; ++step) {
        const auto u = static_cast<VertexId>(v_dist(rng));
        const auto v = static_cast<VertexId>(v_dist(rng));
        const int64_t w = w_dist(rng);
        dyn_mst.insert_edge(u, v, w);
        all_edges.emplace_back(u, v, w);
        if (step % 25 == 24) {
            verify_against_kruskal();
        }
    }
    EXPECT_EQ(dyn_mst.components_count(), 1U);
}

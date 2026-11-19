#include "algoat/graph/max_flow.hpp"
#include "algoat/graph/network_simplex.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using algoat::graph::NetworkSimplex;
using Solver = NetworkSimplex<int64_t, int64_t>;
using ProblemType = Solver::ProblemType;

TEST(NetworkSimplexTest, EmptyGraph) {
    Solver ns(0);
    EXPECT_EQ(ns.num_vertices(), 0u);
    EXPECT_EQ(ns.num_edges(), 0u);
    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 0);
    EXPECT_EQ(ns.flow(0), 0);
    EXPECT_EQ(ns.potential(0), 0);
}

TEST(NetworkSimplexTest, NoEdgesZeroSupplyOptimal) {
    Solver ns(4);
    EXPECT_EQ(ns.num_vertices(), 4u);
    EXPECT_EQ(ns.num_edges(), 0u);
    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 0);
}

TEST(NetworkSimplexTest, NoEdgesNonZeroSupplyInfeasible) {
    Solver ns(3);
    ns.set_supply(0, 5);
    ns.set_supply(2, -5);
    EXPECT_EQ(ns.run(), ProblemType::Infeasible);
}

TEST(NetworkSimplexTest, UnbalancedSupplyInfeasible) {
    Solver ns(3);
    ns.add_edge(0, 1, 10, 1);
    ns.add_edge(1, 2, 10, 1);
    ns.set_supply(0, 10);
    ns.set_supply(2, -9);
    EXPECT_EQ(ns.run(), ProblemType::Infeasible);
}

TEST(NetworkSimplexTest, LowerBoundExceedsCapacityInfeasible) {
    Solver ns(2);
    ns.add_edge(0, 1, 5, 2, /*lower_bound=*/6);
    EXPECT_EQ(ns.run(), ProblemType::Infeasible);
}

TEST(NetworkSimplexTest, NegativeCapacityWithZeroLowerInfeasible) {
    Solver ns(2);
    ns.add_edge(0, 1, -1, 2);
    EXPECT_EQ(ns.run(), ProblemType::Infeasible);
}

TEST(NetworkSimplexTest, OutOfBoundsThrows) {
    Solver ns(3);
    EXPECT_THROW(ns.add_edge(3, 0, 10, 1), std::out_of_range);
    EXPECT_THROW(ns.add_edge(0, 3, 10, 1), std::out_of_range);
    EXPECT_THROW(ns.set_supply(3, 5), std::out_of_range);

    Solver empty(0);
    EXPECT_THROW(empty.add_edge(0, 0, 1, 1), std::out_of_range);
    EXPECT_THROW(empty.set_supply(0, 0), std::out_of_range);
}

TEST(NetworkSimplexTest, SingleEdgeOptimal) {
    Solver ns(2);
    ns.add_edge(0, 1, 10, 7);
    ns.set_supply(0, 6);
    ns.set_supply(1, -6);

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 42);
    EXPECT_EQ(ns.flow(0), 6);
}

TEST(NetworkSimplexTest, DiamondCheapestPathRouting) {
    Solver ns(4);
    ns.add_edge(0, 1, 10, 2); // e0: flow 5
    ns.add_edge(1, 3, 3, 2);  // e1: flow 3
    ns.add_edge(0, 2, 10, 5); // e2: flow 0
    ns.add_edge(2, 3, 10, 3); // e3: flow 2
    ns.add_edge(1, 2, 2, 1);  // e4: flow 2
    ns.set_supply(0, 5);
    ns.set_supply(3, -5);

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 24);
    EXPECT_EQ(ns.flow(0), 5);
    EXPECT_EQ(ns.flow(1), 3);
    EXPECT_EQ(ns.flow(2), 0);
    EXPECT_EQ(ns.flow(3), 2);
    EXPECT_EQ(ns.flow(4), 2);
}

TEST(NetworkSimplexTest, MultiSourceMultiSinkTransportation) {
    Solver ns(5);
    ns.set_supply(0, 4);
    ns.set_supply(1, 6);
    ns.set_supply(2, -3);
    ns.set_supply(3, -4);
    ns.set_supply(4, -3);

    ns.add_edge(0, 2, 10, 2);
    ns.add_edge(0, 3, 10, 4);
    ns.add_edge(0, 4, 10, 5);
    ns.add_edge(1, 2, 10, 3);
    ns.add_edge(1, 3, 10, 1);
    ns.add_edge(1, 4, 10, 2);

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 19);
    const int64_t f02 = ns.flow(0);
    const int64_t f03 = ns.flow(1);
    const int64_t f04 = ns.flow(2);
    const int64_t f12 = ns.flow(3);
    const int64_t f13 = ns.flow(4);
    const int64_t f14 = ns.flow(5);
    for (int64_t f : {f02, f03, f04, f12, f13, f14}) {
        EXPECT_GE(f, 0);
        EXPECT_LE(f, 10);
    }
    EXPECT_EQ(f02 + f03 + f04, 4);
    EXPECT_EQ(f12 + f13 + f14, 6);
    EXPECT_EQ(f02 + f12, 3);
    EXPECT_EQ(f03 + f13, 4);
    EXPECT_EQ(f04 + f14, 3);
}

TEST(NetworkSimplexTest, BipartiteMinWeightAssignment4x4) {
    Solver ns(8);
    const int64_t cost[4][4] = {
        {9, 2, 7, 8},
        {6, 4, 3, 7},
        {5, 8, 1, 8},
        {7, 6, 9, 4},
    };
    for (std::size_t i = 0; i < 4; ++i) {
        ns.set_supply(i, 1);
        ns.set_supply(4 + i, -1);
        for (std::size_t j = 0; j < 4; ++j) {
            ns.add_edge(i, 4 + j, 1, cost[i][j]);
        }
    }

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 13);
    for (std::size_t i = 0; i < 4; ++i) {
        int64_t assigned = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            const int64_t f = ns.flow(i * 4 + j);
            EXPECT_TRUE(f == 0 || f == 1);
            assigned += f;
        }
        EXPECT_EQ(assigned, 1);
    }
}

TEST(NetworkSimplexTest, MultiplePathsCapacitatedSplit) {
    Solver ns(5);
    ns.set_supply(0, 10);
    ns.set_supply(4, -10);

    ns.add_edge(0, 1, 4, 1);
    ns.add_edge(1, 4, 4, 1);
    ns.add_edge(0, 2, 4, 2);
    ns.add_edge(2, 4, 4, 2);
    ns.add_edge(0, 3, 5, 3);
    ns.add_edge(3, 4, 5, 3);

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 36);
    EXPECT_EQ(ns.flow(0), 4);
    EXPECT_EQ(ns.flow(2), 4);
    EXPECT_EQ(ns.flow(4), 2);
}

TEST(NetworkSimplexTest, PositiveLowerBoundsForceFlow) {
    Solver ns(3);
    ns.set_supply(0, 5);
    ns.set_supply(2, -5);

    ns.add_edge(0, 2, 10, 1, 0);
    ns.add_edge(0, 1, 10, 10, 3);
    ns.add_edge(1, 2, 10, 10, 0);

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 62);
    EXPECT_EQ(ns.flow(0), 2);
    EXPECT_EQ(ns.flow(1), 3);
    EXPECT_EQ(ns.flow(2), 3);
}

TEST(NetworkSimplexTest, NegativeLowerBounds) {
    Solver ns(2);
    ns.set_supply(0, -3);
    ns.set_supply(1, 3);
    ns.add_edge(0, 1, 10, 5, /*lower_bound=*/-5);

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), -15);
    EXPECT_EQ(ns.flow(0), -3);

    // Combined with an anti-parallel edge pushing optimal flow onto positive domain
    Solver ns2(2);
    ns2.set_supply(0, -3);
    ns2.set_supply(1, 3);
    ns2.add_edge(0, 1, 4, -10, /*lower_bound=*/-5);
    ns2.add_edge(1, 0, 6, 1, /*lower_bound=*/0);

    EXPECT_EQ(ns2.run(), ProblemType::Optimal);
    EXPECT_EQ(ns2.total_cost(), -24);
    EXPECT_EQ(ns2.flow(0), 3);
    EXPECT_EQ(ns2.flow(1), 6);
}

TEST(NetworkSimplexTest, NegativeEdgeCostsFiniteCapacity) {
    // Acyclic negative costs
    Solver dag(3);
    dag.set_supply(0, 4);
    dag.set_supply(2, -4);
    dag.add_edge(0, 1, 5, -4);
    dag.add_edge(1, 2, 5, -3);
    dag.add_edge(0, 2, 5, -2);

    EXPECT_EQ(dag.run(), ProblemType::Optimal);
    EXPECT_EQ(dag.total_cost(), -28);
    EXPECT_EQ(dag.flow(0), 4);
    EXPECT_EQ(dag.flow(1), 4);
    EXPECT_EQ(dag.flow(2), 0);

    // Negative-cost cycle with zero external supplies saturates around bottleneck
    Solver cyc(3);
    cyc.add_edge(0, 1, 3, -5);
    cyc.add_edge(1, 2, 5, -2);
    cyc.add_edge(2, 0, 4, 1);

    EXPECT_EQ(cyc.run(), ProblemType::Optimal);
    EXPECT_EQ(cyc.total_cost(), -18);
    EXPECT_EQ(cyc.flow(0), 3);
    EXPECT_EQ(cyc.flow(1), 3);
    EXPECT_EQ(cyc.flow(2), 3);
}

TEST(NetworkSimplexTest, ParallelAndAntiParallelEdges) {
    Solver ns(2);
    ns.set_supply(0, 6);
    ns.set_supply(1, -6);

    ns.add_edge(0, 1, 3, 2);
    ns.add_edge(0, 1, 4, 5);
    ns.add_edge(0, 1, 10, 10);
    ns.add_edge(1, 0, 5, 1);

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 21);
    EXPECT_EQ(ns.flow(0), 3);
    EXPECT_EQ(ns.flow(1), 3);
    EXPECT_EQ(ns.flow(2), 0);
    EXPECT_EQ(ns.flow(3), 0);
}

TEST(NetworkSimplexTest, SelfLoopsPositiveAndNegativeCost) {
    Solver ns(2);
    ns.set_supply(0, 3);
    ns.set_supply(1, -3);

    ns.add_edge(0, 0, 10, 5, /*lower_bound=*/2); // fixed at lower bound 2 -> +10
    ns.add_edge(1, 1, 7, -3, /*lower_bound=*/0); // saturates to upper bound 7 -> -21
    ns.add_edge(0, 1, 5, 4, /*lower_bound=*/0);  // carries 3 -> +12

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 1);
    EXPECT_EQ(ns.flow(0), 2);
    EXPECT_EQ(ns.flow(1), 7);
    EXPECT_EQ(ns.flow(2), 3);
}

TEST(NetworkSimplexTest, ZeroCapacityAndFixedLowerEqualsUpper) {
    Solver ns(2);
    ns.set_supply(0, 6);
    ns.set_supply(1, -6);

    ns.add_edge(0, 1, 0, -100, 0); // zero capacity -> flow 0
    ns.add_edge(0, 1, 4, 5, 4);    // fixed at 4 -> cost 20
    ns.add_edge(0, 1, 10, 1, 0);   // remaining 2 -> cost 2

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 22);
    EXPECT_EQ(ns.flow(0), 0);
    EXPECT_EQ(ns.flow(1), 4);
    EXPECT_EQ(ns.flow(2), 2);
}

TEST(NetworkSimplexTest, DisconnectedFeasibleComponents) {
    Solver ns(4);
    ns.set_supply(0, 4);
    ns.set_supply(1, -4);
    ns.set_supply(2, 2);
    ns.set_supply(3, -2);

    ns.add_edge(0, 1, 5, 3);
    ns.add_edge(2, 3, 5, -2);

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 8);
    EXPECT_EQ(ns.flow(0), 4);
    EXPECT_EQ(ns.flow(1), 2);
}

TEST(NetworkSimplexTest, RunIdempotencyAndSubsequentMutation) {
    Solver ns(3);
    ns.set_supply(0, 5);
    ns.set_supply(2, -5);
    ns.add_edge(0, 1, 10, 4, /*lower_bound=*/2);
    ns.add_edge(1, 2, 10, 4);
    ns.add_edge(0, 2, 10, 12);

    ASSERT_EQ(ns.run(), ProblemType::Optimal);
    const int64_t cost1 = ns.total_cost();
    const int64_t f0_1 = ns.flow(0);
    const int64_t f2_1 = ns.flow(2);
    // 2 units via 0->1->2 cost 8/unit; remaining 3 units via 0->1->2 also cost 8/unit < 12 -> 40.
    EXPECT_EQ(cost1, 40);

    // Second call must produce identical results (no double-shifting of lower_bound!)
    ASSERT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 40);
    EXPECT_EQ(ns.flow(0), f0_1);
    EXPECT_EQ(ns.flow(2), f2_1);

    // Mutate graph by adding a direct cheaper edge: 2 still forced via 0->1->2 (cost 16), 3 via
    // 0->2 (cost 3) -> total 19
    ns.add_edge(0, 2, 10, 1);
    ASSERT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 19);
    EXPECT_EQ(ns.flow(0), 2);
    EXPECT_EQ(ns.flow(3), 3);
}

TEST(NetworkSimplexTest, InfeasibleCapacityCut) {
    Solver ns(2);
    ns.set_supply(0, 6);
    ns.set_supply(1, -6);
    ns.add_edge(0, 1, 5, 1);
    EXPECT_EQ(ns.run(), ProblemType::Infeasible);
}

TEST(NetworkSimplexTest, InfeasibleLowerBoundCirculation) {
    Solver ns(2);
    ns.add_edge(0, 1, 10, 1, /*lower_bound=*/6);
    ns.add_edge(1, 0, 4, 1, /*lower_bound=*/0);
    EXPECT_EQ(ns.run(), ProblemType::Infeasible);
}

TEST(NetworkSimplexTest, UnboundedNegativeCycleInfiniteCapacity) {
    Solver ns(3);
    constexpr auto kInf = Solver::kInfiniteCapacity;
    ns.add_edge(0, 1, kInf, -3);
    ns.add_edge(1, 2, kInf, -2);
    ns.add_edge(2, 0, kInf, 1);
    EXPECT_EQ(ns.run(), ProblemType::Unbounded);
}

TEST(NetworkSimplexTest, UnboundedNegativeSelfLoopWhenFeasible) {
    Solver ns(2);
    ns.set_supply(0, 3);
    ns.set_supply(1, -3);
    ns.add_edge(0, 0, Solver::kInfiniteCapacity, -1);
    ns.add_edge(0, 1, 5, 2);
    EXPECT_EQ(ns.run(), ProblemType::Unbounded);
}

TEST(NetworkSimplexTest, InfeasiblePrecedesUnboundedNegativeCycle) {
    Solver ns(4);
    ns.set_supply(0, 5);
    ns.set_supply(1, -5);
    ns.add_edge(0, 1, 1, 1); // bottleneck < 5 -> Infeasible

    // Disjoint component with infinite negative-cost cycle
    ns.add_edge(2, 3, Solver::kInfiniteCapacity, -5);
    ns.add_edge(3, 2, Solver::kInfiniteCapacity, 1);

    EXPECT_EQ(ns.run(), ProblemType::Infeasible);
}

TEST(NetworkSimplexTest, InfeasiblePrecedesUnboundedNegativeSelfLoop) {
    Solver ns(2);
    ns.set_supply(0, 5);
    ns.set_supply(1, -5);
    ns.add_edge(0, 1, 1, 1);
    ns.add_edge(0, 0, Solver::kInfiniteCapacity, -10);

    EXPECT_EQ(ns.run(), ProblemType::Infeasible);
}

TEST(NetworkSimplexTest, DegenerateZeroCostMeshNoCycling) {
    constexpr std::size_t n = 4;
    Solver ns(n * n);
    ns.set_supply(0, 10);
    ns.set_supply(n * n - 1, -10);

    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            const std::size_t u = r * n + c;
            if (r + 1 < n) {
                const std::size_t d = (r + 1) * n + c;
                ns.add_edge(u, d, 10, 0);
                ns.add_edge(d, u, 10, 0);
            }
            if (c + 1 < n) {
                const std::size_t right = r * n + (c + 1);
                ns.add_edge(u, right, 10, 0);
                ns.add_edge(right, u, 10, 0);
            }
        }
    }

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 0);
}

TEST(NetworkSimplexTest, CunninghamDegenerateStallingStress) {
    // Layered degenerate network with tied zero-cost and negative-cost residual loops
    // forcing degenerate pivots across zero-flow tree arcs.
    constexpr std::size_t layers = 12;
    constexpr std::size_t width = 4;
    const std::size_t num_v = layers * width + 2;
    const std::size_t s = num_v - 2;
    const std::size_t t = num_v - 1;

    Solver ns(num_v);
    ns.set_supply(s, 1);
    ns.set_supply(t, -1);

    for (std::size_t w = 0; w < width; ++w) {
        ns.add_edge(s, w, 1, 0);
        ns.add_edge((layers - 1) * width + w, t, 1, 0);
    }
    for (std::size_t l = 0; l + 1 < layers; ++l) {
        for (std::size_t u = 0; u < width; ++u) {
            for (std::size_t v = 0; v < width; ++v) {
                const int64_t cost = (u == 0 && v == 0) ? -1 : 0;
                ns.add_edge(l * width + u, (l + 1) * width + v, 1, cost);
            }
        }
    }

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), -static_cast<int64_t>(layers - 1));
}

TEST(NetworkSimplexTest, Int32WideCostOverflowSafety) {
    using Solver32 = NetworkSimplex<int32_t, int32_t>;
    constexpr std::size_t v = 250;
    Solver32 ns(v);
    ns.set_supply(0, 1);
    ns.set_supply(1, -1);

    // Force art_cost = 1 + 250 * 10'000'000 > INT32_MAX, triggering WideCost path
    ns.add_edge(0, 1, 1, 10'000'000);
    ns.add_edge(0, 1, 1, 5'000'000);
    for (std::size_t i = 1; i + 1 < v; ++i) {
        ns.add_edge(i, i + 1, 1, 1);
    }

    EXPECT_EQ(ns.run(), Solver32::ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 5'000'000);
    EXPECT_EQ(ns.flow(1), 1);
}

TEST(NetworkSimplexTest, Int64ExtremeCostOverflowSafety) {
    Solver ns(4);
    ns.set_supply(0, 2);
    ns.set_supply(3, -2);

    const int64_t high_cost = 100'000'000'000'000'000LL; // 10^17
    const int64_t low_cost = 50'000'000'000'000'000LL;   // 5 * 10^16
    ns.add_edge(0, 1, 5, high_cost);
    ns.add_edge(1, 3, 5, high_cost);
    ns.add_edge(0, 2, 5, low_cost);
    ns.add_edge(2, 3, 5, low_cost);

    EXPECT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), 200'000'000'000'000'000LL);
    EXPECT_EQ(ns.flow(2), 2);
    EXPECT_EQ(ns.flow(3), 2);
}

namespace {

struct TestEdge {
    std::size_t from{0};
    std::size_t to{0};
    int64_t cap{0};
    int64_t cost{0};
    int64_t lower{0};
};

#if defined(__SIZEOF_INT128__)
__extension__ using TestWideInt = __int128;
#else
using TestWideInt = int64_t;
#endif

// Independent O(V + E) primal-dual KKT complementary slackness certifier.
void verify_kkt_optimality(std::size_t num_v, const std::vector<TestEdge>& edges,
                           const std::vector<int64_t>& supplies, const Solver& solver) {
    std::vector<TestWideInt> net_flow(num_v, TestWideInt{0});
    TestWideInt computed_cost{0};

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const TestEdge& e = edges[i];
        const int64_t f = solver.flow(i);
        EXPECT_GE(f, e.lower) << "Edge " << i << " below lower bound";
        EXPECT_LE(f, e.cap) << "Edge " << i << " above upper bound";

        net_flow[e.from] += static_cast<TestWideInt>(f);
        net_flow[e.to] -= static_cast<TestWideInt>(f);
        computed_cost += static_cast<TestWideInt>(f) * static_cast<TestWideInt>(e.cost);

        if (e.from != e.to && e.lower < e.cap) {
            const TestWideInt rc = static_cast<TestWideInt>(e.cost) -
                                   static_cast<TestWideInt>(solver.potential(e.from)) +
                                   static_cast<TestWideInt>(solver.potential(e.to));
            if (f > e.lower) {
                EXPECT_LE(rc, TestWideInt{0})
                    << "Complementary slackness violated (rc > 0 with flow > lower) on edge " << i;
            }
            if (f < e.cap) {
                EXPECT_GE(rc, TestWideInt{0})
                    << "Complementary slackness violated (rc < 0 with flow < upper) on edge " << i;
            }
        }
    }

    for (std::size_t u = 0; u < num_v; ++u) {
        EXPECT_TRUE(net_flow[u] == static_cast<TestWideInt>(supplies[u]))
            << "Flow conservation violated at vertex " << u;
    }

    EXPECT_TRUE(static_cast<TestWideInt>(solver.total_cost()) == computed_cost)
        << "Reported total_cost disagrees with edge-summed cost";
}

// Independent feasibility checker using Dinic's Max-Flow engine.
bool check_feasible_via_dinic(std::size_t num_v, const std::vector<TestEdge>& edges,
                              const std::vector<int64_t>& supplies) {
    int64_t sum = 0;
    std::vector<int64_t> b = supplies;
    for (const int64_t s : supplies) {
        sum += s;
    }
    if (sum != 0) {
        return false;
    }
    for (const auto& e : edges) {
        if (e.lower > e.cap) {
            return false;
        }
        if (e.from != e.to) {
            b[e.from] -= e.lower;
            b[e.to] += e.lower;
        }
    }

    int64_t required = 0;
    for (std::size_t u = 0; u < num_v; ++u) {
        if (b[u] > 0) {
            required += b[u];
        }
    }
    const int64_t safe_inf = required + 1;

    const algoat::graph::VertexId src = static_cast<algoat::graph::VertexId>(num_v);
    const algoat::graph::VertexId snk = static_cast<algoat::graph::VertexId>(num_v + 1);
    algoat::graph::MaxFlowEngine<int64_t> mf(num_v + 2);

    for (const auto& e : edges) {
        if (e.from != e.to && e.lower < e.cap) {
            const int64_t c = (e.cap == Solver::kInfiniteCapacity) ? safe_inf : (e.cap - e.lower);
            mf.add_edge(static_cast<algoat::graph::VertexId>(e.from),
                        static_cast<algoat::graph::VertexId>(e.to), c);
        }
    }

    for (std::size_t u = 0; u < num_v; ++u) {
        if (b[u] > 0) {
            mf.add_edge(src, static_cast<algoat::graph::VertexId>(u), b[u]);
        } else if (b[u] < 0) {
            mf.add_edge(static_cast<algoat::graph::VertexId>(u), snk, -b[u]);
        }
    }

    const auto res = mf.dinic(src, snk);
    return res.max_flow == required;
}

// Independent Successive Shortest Path (Dijkstra + Johnson potentials) oracle for finite
// capacitated networks. Handles negative costs via upfront AMO 9.2 edge saturation/reversal.
std::pair<ProblemType, int64_t> ssp_reference_oracle(std::size_t num_v,
                                                     const std::vector<TestEdge>& edges,
                                                     const std::vector<int64_t>& supplies) {
    int64_t sum = 0;
    std::vector<int64_t> b = supplies;
    for (const int64_t s : supplies) {
        sum += s;
    }
    if (sum != 0) {
        return {ProblemType::Infeasible, 0};
    }

    int64_t base_cost = 0;
    struct ResEdge {
        std::size_t to;
        std::size_t rev;
        int64_t cap;
        int64_t cost;
    };
    std::vector<std::vector<ResEdge>> adj(num_v + 2);
    auto add_res = [&](std::size_t u, std::size_t v, int64_t cap, int64_t cost) {
        const std::size_t fwd = adj[u].size();
        const std::size_t bwd = adj[v].size();
        adj[u].push_back({v, bwd, cap, cost});
        adj[v].push_back({u, fwd, 0, -cost});
    };

    for (const auto& e : edges) {
        if (e.lower > e.cap) {
            return {ProblemType::Infeasible, 0};
        }
        if (e.from == e.to) {
            const int64_t f = (e.cost < 0) ? e.cap : e.lower;
            base_cost += f * e.cost;
            continue;
        }
        base_cost += e.lower * e.cost;
        b[e.from] -= e.lower;
        b[e.to] += e.lower;

        const int64_t rem = e.cap - e.lower;
        if (rem == 0) {
            continue;
        }
        if (e.cost < 0) {
            // Pre-saturate negative-cost arc so all initial residual edges have non-negative cost.
            base_cost += rem * e.cost;
            b[e.from] -= rem;
            b[e.to] += rem;
            add_res(e.to, e.from, rem, -e.cost);
        } else {
            add_res(e.from, e.to, rem, e.cost);
        }
    }

    const std::size_t total_v = num_v + 2;
    const std::size_t src = num_v;
    const std::size_t snk = num_v + 1;
    int64_t required = 0;
    for (std::size_t u = 0; u < num_v; ++u) {
        if (b[u] > 0) {
            add_res(src, u, b[u], 0);
            required += b[u];
        } else if (b[u] < 0) {
            add_res(u, snk, -b[u], 0);
        }
    }

    constexpr int64_t kDistInf = std::numeric_limits<int64_t>::max() / 4;
    std::vector<int64_t> pot(total_v, 0);
    std::vector<int64_t> dist(total_v);
    std::vector<std::size_t> pv(total_v);
    std::vector<std::size_t> pe(total_v);

    using PQItem = std::pair<int64_t, std::size_t>;
    int64_t pushed_total = 0;
    int64_t total_cost = base_cost;

    while (pushed_total < required) {
        std::fill(dist.begin(), dist.end(), kDistInf);
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
        dist[src] = 0;
        pq.push({0, src});

        while (!pq.empty()) {
            const auto [d, u] = pq.top();
            pq.pop();
            if (d > dist[u]) {
                continue;
            }
            if (u == snk) {
                break;
            }
            for (std::size_t idx = 0; idx < adj[u].size(); ++idx) {
                const ResEdge& edge = adj[u][idx];
                if (edge.cap > 0) {
                    const int64_t nd = d + edge.cost + pot[u] - pot[edge.to];
                    if (nd < dist[edge.to]) {
                        dist[edge.to] = nd;
                        pv[edge.to] = u;
                        pe[edge.to] = idx;
                        pq.push({nd, edge.to});
                    }
                }
            }
        }

        if (dist[snk] >= kDistInf) {
            break;
        }

        const int64_t snk_dist = dist[snk];
        for (std::size_t u = 0; u < total_v; ++u) {
            pot[u] += (dist[u] < snk_dist) ? dist[u] : snk_dist;
        }

        int64_t aug = required - pushed_total;
        for (std::size_t v = snk; v != src; v = pv[v]) {
            aug = std::min(aug, adj[pv[v]][pe[v]].cap);
        }
        for (std::size_t v = snk; v != src; v = pv[v]) {
            ResEdge& fwd = adj[pv[v]][pe[v]];
            fwd.cap -= aug;
            adj[v][fwd.rev].cap += aug;
        }
        pushed_total += aug;
        total_cost += aug * (pot[snk] - pot[src]);
    }

    if (pushed_total < required) {
        return {ProblemType::Infeasible, 0};
    }
    return {ProblemType::Optimal, total_cost};
}

struct NetgenParams {
    std::size_t num_v{0};
    std::size_t num_edges{0};
    std::size_t num_sources{0};
    std::size_t num_sinks{0};
    int64_t avg_supply{1000};
    int64_t min_cost{1};
    int64_t max_cost{10000};
    int64_t min_cap{1};
    int64_t max_cap{1000};
    uint64_t seed{42};
};

// Klingman-Napier-Stutz (1974) NETGEN-style feasible capacitated network generator.
void generate_netgen_instance(const NetgenParams& p, std::vector<TestEdge>& edges,
                              std::vector<int64_t>& supplies) {
    edges.clear();
    supplies.assign(p.num_v, 0);

    std::mt19937_64 rng(p.seed);
    std::uniform_int_distribution<int64_t> cost_dist(p.min_cost, p.max_cost);
    std::uniform_int_distribution<int64_t> cap_dist(p.min_cap, p.max_cap);

    const int64_t total_supply = static_cast<int64_t>(p.num_sources) * p.avg_supply;
    for (std::size_t i = 0; i < p.num_sources; ++i) {
        supplies[i] = p.avg_supply;
    }
    for (std::size_t i = 0; i < p.num_sources; ++i) {
        const std::size_t j = rng() % p.num_sources;
        if (i != j && supplies[i] > 1) {
            const int64_t shift = static_cast<int64_t>(rng() % static_cast<uint64_t>(supplies[i]));
            supplies[i] -= shift;
            supplies[j] += shift;
        }
    }

    const std::size_t sink_base = p.num_v - p.num_sinks;
    int64_t rem_demand = total_supply;
    for (std::size_t i = 0; i + 1 < p.num_sinks; ++i) {
        const int64_t share = total_supply / static_cast<int64_t>(p.num_sinks);
        const int64_t jitter =
            static_cast<int64_t>(rng() % static_cast<uint64_t>(std::max<int64_t>(1, share / 2)));
        const int64_t d = std::min(rem_demand, share / 2 + jitter);
        supplies[sink_base + i] = -d;
        rem_demand -= d;
    }
    supplies[sink_base + p.num_sinks - 1] = -rem_demand;

    // Feasibility skeleton chains from sources through transshipment nodes to sinks
    const std::size_t trans_count = p.num_v - p.num_sources - p.num_sinks;
    std::vector<std::size_t> trans_nodes(trans_count);
    for (std::size_t i = 0; i < trans_count; ++i) {
        trans_nodes[i] = p.num_sources + i;
    }
    std::shuffle(trans_nodes.begin(), trans_nodes.end(), rng);

    std::size_t cursor = 0;
    std::vector<std::size_t> source_tail(p.num_sources);
    for (std::size_t s = 0; s < p.num_sources; ++s) {
        std::size_t prev = s;
        const std::size_t chain_len = (trans_count - cursor) / (p.num_sources - s);
        for (std::size_t k = 0; k < chain_len; ++k) {
            const std::size_t next = trans_nodes[cursor++];
            const int64_t c = (k % 3 == 0) ? p.max_cost : cost_dist(rng);
            edges.push_back({prev, next, std::max(supplies[s], cap_dist(rng)), c, 0});
            prev = next;
        }
        source_tail[s] = prev;
    }

    // Two-pointer supply-demand skeleton matching guarantees a feasible flow exists
    std::size_t s_idx = 0;
    std::size_t t_idx = 0;
    int64_t rem_s = supplies[0];
    int64_t rem_t = -supplies[sink_base];
    while (s_idx < p.num_sources && t_idx < p.num_sinks) {
        const int64_t flow_needed = std::min(rem_s, rem_t);
        if (flow_needed > 0) {
            edges.push_back({source_tail[s_idx], sink_base + t_idx,
                             std::max(flow_needed, cap_dist(rng)), p.max_cost, 0});
        }
        rem_s -= flow_needed;
        rem_t -= flow_needed;
        if (rem_s == 0 && ++s_idx < p.num_sources) {
            rem_s = supplies[s_idx];
        }
        if (rem_t == 0 && ++t_idx < p.num_sinks) {
            rem_t = -supplies[sink_base + t_idx];
        }
    }

    while (edges.size() < p.num_edges) {
        const std::size_t u = rng() % sink_base; // exclude pure sinks as tails
        const std::size_t v =
            p.num_sources + (rng() % (p.num_v - p.num_sources)); // exclude pure sources
        if (u != v) {
            edges.push_back({u, v, cap_dist(rng), cost_dist(rng), 0});
        }
    }
}

} // namespace

TEST(NetworkSimplexTest, CanonicalAhujaMagnantiOrlinTextbookInstance) {
    // Classic 4-node network (AMO Ch. 9/11 style):
    // b = [4, 0, -2, -2]; edges: 0->1(cap 4, cost 2), 0->2(cap 2, cost 2),
    // 1->2(cap 2, cost 1), 1->3(cap 3, cost 3), 2->3(cap 5, cost 1)
    const std::vector<TestEdge> edges = {
        {0, 1, 4, 2, 0}, {0, 2, 2, 2, 0}, {1, 2, 2, 1, 0}, {1, 3, 3, 3, 0}, {2, 3, 5, 1, 0},
    };
    const std::vector<int64_t> supplies = {4, 0, -2, -2};

    Solver ns(4);
    for (std::size_t i = 0; i < 4; ++i) {
        ns.set_supply(i, supplies[i]);
    }
    for (const auto& e : edges) {
        ns.add_edge(e.from, e.to, e.cap, e.cost, e.lower);
    }

    ASSERT_EQ(ns.run(), ProblemType::Optimal);
    // 2 via 0->2 (cost 4), 2 via 0->1->2->3 (cost 2*(2+1+1)=8) -> total 12
    EXPECT_EQ(ns.total_cost(), 12);
    verify_kkt_optimality(4, edges, supplies, ns);

    const auto [ref_status, ref_cost] = ssp_reference_oracle(4, edges, supplies);
    EXPECT_EQ(ref_status, ProblemType::Optimal);
    EXPECT_EQ(ns.total_cost(), ref_cost);
}

TEST(NetworkSimplexTest, RandomizedDifferentialAgainstSSPAndDinic) {
    std::mt19937_64 rng(0x51A9F10DULL);
    std::uniform_int_distribution<std::size_t> v_dist(2, 20);
    std::uniform_int_distribution<std::size_t> e_dist(1, 55);
    std::uniform_int_distribution<int64_t> cost_dist(-100, 100);
    std::uniform_int_distribution<int64_t> lower_dist(-10, 10);
    std::uniform_int_distribution<int64_t> span_dist(0, 30);

    for (std::size_t iter = 0; iter < 1500; ++iter) {
        const std::size_t num_v = v_dist(rng);
        const std::size_t num_e = e_dist(rng);

        std::vector<int64_t> supplies(num_v, 0);
        if (iter % 15 == 0) {
            // Unbalanced supplies test
            supplies[0] = 5;
            supplies[1] = -4;
        } else {
            for (std::size_t k = 0; k < num_v; ++k) {
                const std::size_t u = rng() % num_v;
                const std::size_t v = rng() % num_v;
                const int64_t amt = static_cast<int64_t>((rng() % 20) + 1);
                supplies[u] += amt;
                supplies[v] -= amt;
            }
        }

        std::vector<TestEdge> edges;
        edges.reserve(num_e);
        Solver ns(num_v);
        for (std::size_t u = 0; u < num_v; ++u) {
            ns.set_supply(u, supplies[u]);
        }
        for (std::size_t i = 0; i < num_e; ++i) {
            const std::size_t u = rng() % num_v;
            const std::size_t v = rng() % num_v;
            const int64_t lb = (iter % 3 == 0) ? lower_dist(rng) : 0;
            const int64_t cap = lb + span_dist(rng);
            const int64_t c = cost_dist(rng);
            edges.push_back({u, v, cap, c, lb});
            ns.add_edge(u, v, cap, c, lb);
        }

        const ProblemType status = ns.run();
        const auto [oracle_status, oracle_cost] = ssp_reference_oracle(num_v, edges, supplies);
        const bool dinic_feasible = check_feasible_via_dinic(num_v, edges, supplies);

        ASSERT_EQ(status, oracle_status) << "Status mismatch at iter " << iter;
        ASSERT_EQ(status == ProblemType::Optimal, dinic_feasible)
            << "Feasibility mismatch with Dinic at iter " << iter;

        if (status == ProblemType::Optimal) {
            ASSERT_EQ(ns.total_cost(), oracle_cost) << "Cost mismatch at iter " << iter;
            verify_kkt_optimality(num_v, edges, supplies, ns);
        }
    }
}

TEST(NetworkSimplexTest, DimacsNetgenOptimalityCertification) {
    struct NetgenConfig {
        std::size_t num_v;
        std::size_t num_e;
        std::size_t sources;
        std::size_t sinks;
        bool check_ssp;
    };
    const std::vector<NetgenConfig> configs = {
        {128, 1024, 11, 11, true},   // NETGEN-8
        {256, 4096, 16, 16, true},   // NETGEN-SR
        {1024, 8192, 32, 32, false}, // NETGEN-8 medium-large
        {1024, 16384, 32, 32, false} // NETGEN-SR-like
    };

    for (std::size_t c = 0; c < configs.size(); ++c) {
        const auto& cfg = configs[c];
        std::vector<TestEdge> edges;
        std::vector<int64_t> supplies;
        generate_netgen_instance(NetgenParams{cfg.num_v, cfg.num_e, cfg.sources, cfg.sinks, 1000, 1,
                                              10000, 1, 1000, 100 + c},
                                 edges, supplies);

        Solver ns(cfg.num_v);
        for (std::size_t u = 0; u < cfg.num_v; ++u) {
            ns.set_supply(u, supplies[u]);
        }
        for (const auto& e : edges) {
            ns.add_edge(e.from, e.to, e.cap, e.cost, e.lower);
        }

        ASSERT_EQ(ns.run(), ProblemType::Optimal);
        verify_kkt_optimality(cfg.num_v, edges, supplies, ns);

        if (cfg.check_ssp) {
            const auto [oracle_status, oracle_cost] =
                ssp_reference_oracle(cfg.num_v, edges, supplies);
            ASSERT_EQ(oracle_status, ProblemType::Optimal);
            EXPECT_EQ(ns.total_cost(), oracle_cost);
        }
    }
}

TEST(NetworkSimplexTest, FiniteCapacityExceedingInt64MaxViaNegativeLowerBound) {
    Solver ns(2);
    constexpr int64_t kCap0 = Solver::kInfiniteCapacity - 5;
    constexpr int64_t kCap1 = Solver::kInfiniteCapacity - 1;
    // Negative-cost cycle where (0 -> 1) has finite upper bound kCap0 and negative lower bound
    // so shifted span (kCap0 + 10) exceeds INT64_MAX yet remains strictly finite.
    ns.add_edge(0, 1, kCap0, -1, /*lower_bound=*/-10);
    ns.add_edge(1, 0, kCap1, 0, /*lower_bound=*/0);

    ASSERT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.flow(0), kCap0);
    EXPECT_EQ(ns.flow(1), kCap0);
    EXPECT_EQ(ns.total_cost(), -kCap0);
}

TEST(NetworkSimplexTest, ShiftedSupplyExceedingInt64MaxViaLowerBounds) {
    Solver ns(2);
    constexpr int64_t kFiveE18 = 5'000'000'000'000'000'000LL;
    constexpr int64_t kSixE18 = 6'000'000'000'000'000'000LL;
    ns.set_supply(0, kFiveE18);
    ns.set_supply(1, -kFiveE18);

    // Lower bound -5e18 shifts node 0 supply to +1e19 (> INT64_MAX), while actual flows fit
    // cleanly.
    ns.add_edge(0, 1, 0, 1, /*lower_bound=*/-kFiveE18);
    ns.add_edge(0, 1, kSixE18, 2, /*lower_bound=*/0);

    ASSERT_EQ(ns.run(), ProblemType::Optimal);
    EXPECT_EQ(ns.flow(0), 0);
    EXPECT_EQ(ns.flow(1), kFiveE18);
}

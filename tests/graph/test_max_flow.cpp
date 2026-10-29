#include "algoat/graph/max_flow.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <unordered_set>
#include <vector>

using namespace algoat::graph;

namespace {

// Helper to verify flow conservation and capacity bounds
template <typename FlowType, typename ExpectedType = FlowType>
void verify_flow_invariants(std::size_t num_vertices, VertexId source, VertexId sink,
                            const MaxFlowResult<FlowType>& result, ExpectedType expected_flow_in) {
    FlowType expected_flow = static_cast<FlowType>(expected_flow_in);
    EXPECT_EQ(result.max_flow, expected_flow);

    std::vector<FlowType> net_flow(num_vertices, 0);

    for (const auto& e : result.residual_edges) {
        // Capacity bound
        EXPECT_GE(e.flow, 0);
        EXPECT_LE(e.flow, e.capacity);

        net_flow[e.from] -= e.flow;
        net_flow[e.to] += e.flow;
    }

    if (source != sink && num_vertices > 0 && source < num_vertices && sink < num_vertices) {
        // Source outflow
        EXPECT_EQ(-net_flow[source], expected_flow);
        // Sink inflow
        EXPECT_EQ(net_flow[sink], expected_flow);

        // Conservation on all internal nodes
        for (VertexId v = 0; v < num_vertices; ++v) {
            if (v != source && v != sink) {
                EXPECT_EQ(net_flow[v], 0) << "Flow conservation violated at vertex " << v;
            }
        }
    }

    // Min-cut verification
    std::unordered_set<VertexId> s_set(result.min_cut_source_set.begin(),
                                       result.min_cut_source_set.end());
    if (source < num_vertices && sink < num_vertices && source != sink) {
        EXPECT_TRUE(s_set.contains(source));
        EXPECT_FALSE(s_set.contains(sink));

        FlowType cut_capacity = 0;
        for (const auto& e : result.residual_edges) {
            if (s_set.contains(e.from) && !s_set.contains(e.to)) {
                cut_capacity += e.capacity;
            }
        }
        EXPECT_EQ(cut_capacity, result.max_flow) << "Min-cut capacity must equal max-flow";
    }
}

} // namespace

TEST(MaxFlowTest, EmptyGraph) {
    MaxFlowEngine<int64_t> engine(0);
    EXPECT_EQ(engine.num_vertices(), 0u);
    EXPECT_EQ(engine.num_edges(), 0u);

    auto res_dinic = engine.dinic(0, 0);
    EXPECT_EQ(res_dinic.max_flow, 0);

    auto res_pr = engine.push_relabel(0, 0);
    EXPECT_EQ(res_pr.max_flow, 0);
}

TEST(MaxFlowTest, SourceEqualsSink) {
    MaxFlowEngine<int64_t> engine(3);
    engine.add_edge(0, 1, 10);
    engine.add_edge(1, 2, 10);

    auto res_dinic = engine.dinic(0, 0);
    EXPECT_EQ(res_dinic.max_flow, 0);

    auto res_pr = engine.push_relabel(1, 1);
    EXPECT_EQ(res_pr.max_flow, 0);
}

TEST(MaxFlowTest, DisconnectedGraph) {
    MaxFlowEngine<int64_t> engine(4);
    engine.add_edge(0, 1, 10);
    engine.add_edge(2, 3, 10);

    auto res_dinic = engine.dinic(0, 3);
    EXPECT_EQ(res_dinic.max_flow, 0);
    verify_flow_invariants(4, 0, 3, res_dinic, 0);

    auto res_pr = engine.push_relabel(0, 3);
    EXPECT_EQ(res_pr.max_flow, 0);
    verify_flow_invariants(4, 0, 3, res_pr, 0);
}

TEST(MaxFlowTest, SingleEdge) {
    MaxFlowEngine<int64_t> engine(2);
    engine.add_edge(0, 1, 42);

    auto res_dinic = engine.dinic(0, 1);
    verify_flow_invariants(2, 0, 1, res_dinic, 42);

    auto res_pr = engine.push_relabel(0, 1);
    verify_flow_invariants(2, 0, 1, res_pr, 42);
}

TEST(MaxFlowTest, ParallelAndSelfLoopEdges) {
    MaxFlowEngine<int64_t> engine(3);
    engine.add_edge(0, 0, 100); // self-loop
    engine.add_edge(0, 1, 15);
    engine.add_edge(0, 1, 25); // parallel edge
    engine.add_edge(1, 2, 50);
    engine.add_edge(2, 2, 50); // self loop

    auto res_dinic = engine.dinic(0, 2);
    verify_flow_invariants(3, 0, 2, res_dinic, 40);

    auto res_pr = engine.push_relabel(0, 2);
    verify_flow_invariants(3, 0, 2, res_pr, 40);
}

TEST(MaxFlowTest, AntiParallelEdges) {
    MaxFlowEngine<int64_t> engine(3);
    // 0 -> 1 (10) and 1 -> 0 (5)
    engine.add_edge(0, 1, 10);
    engine.add_edge(1, 0, 5);
    engine.add_edge(1, 2, 8);

    auto res_dinic = engine.dinic(0, 2);
    verify_flow_invariants(3, 0, 2, res_dinic, 8);

    auto res_pr = engine.push_relabel(0, 2);
    verify_flow_invariants(3, 0, 2, res_pr, 8);
}

TEST(MaxFlowTest, DiamondNetwork) {
    // Classic diamond graph:
    // 0 -> 1: 10
    // 0 -> 2: 10
    // 1 -> 2: 1
    // 1 -> 3: 10
    // 2 -> 3: 10
    MaxFlowEngine<int64_t> engine(4);
    engine.add_edge(0, 1, 10);
    engine.add_edge(0, 2, 10);
    engine.add_edge(1, 2, 1);
    engine.add_edge(1, 3, 10);
    engine.add_edge(2, 3, 10);

    auto res_dinic = engine.dinic(0, 3);
    verify_flow_invariants(4, 0, 3, res_dinic, 20);

    auto res_pr = engine.push_relabel(0, 3);
    verify_flow_invariants(4, 0, 3, res_pr, 20);
}

TEST(MaxFlowTest, BipartiteMatchingUnitNetwork) {
    // Source: 0, Sink: 5
    // Left: 1, 2. Right: 3, 4
    // 1 -> 3, 1 -> 4, 2 -> 3
    MaxFlowEngine<int64_t> engine(6);
    engine.add_edge(0, 1, 1);
    engine.add_edge(0, 2, 1);
    engine.add_edge(1, 3, 1);
    engine.add_edge(1, 4, 1);
    engine.add_edge(2, 3, 1);
    engine.add_edge(3, 5, 1);
    engine.add_edge(4, 5, 1);

    auto res_dinic = engine.dinic(0, 5);
    verify_flow_invariants(6, 0, 5, res_dinic, 2);

    auto res_pr = engine.push_relabel(0, 5);
    verify_flow_invariants(6, 0, 5, res_pr, 2);
}

TEST(MaxFlowTest, Large64BitCapacities) {
    MaxFlowEngine<int64_t> engine(4);
    const int64_t cap1 = 1000000000000LL; // 10^12
    const int64_t cap2 = 2000000000000LL; // 2 * 10^12
    engine.add_edge(0, 1, cap1);
    engine.add_edge(0, 2, cap2);
    engine.add_edge(1, 3, cap1);
    engine.add_edge(2, 3, cap2);

    auto res_dinic = engine.dinic(0, 3);
    verify_flow_invariants(4, 0, 3, res_dinic, cap1 + cap2);

    auto res_pr = engine.push_relabel(0, 3);
    verify_flow_invariants(4, 0, 3, res_pr, cap1 + cap2);
}

TEST(MaxFlowTest, OutOfBoundsThrows) {
    MaxFlowEngine<int64_t> engine(4);
    EXPECT_THROW(engine.add_edge(4, 0, 10), std::out_of_range);
    EXPECT_THROW(engine.add_edge(0, 4, 10), std::out_of_range);
}

TEST(MaxFlowTest, EngineReusabilityAndIdempotency) {
    MaxFlowEngine<int64_t> engine(4);
    engine.add_edge(0, 1, 10);
    engine.add_edge(0, 2, 10);
    engine.add_edge(1, 2, 1);
    engine.add_edge(1, 3, 10);
    engine.add_edge(2, 3, 10);

    // Run Dinic multiple times
    auto d1 = engine.dinic(0, 3);
    auto d2 = engine.dinic(0, 3);
    EXPECT_EQ(d1.max_flow, d2.max_flow);

    // Run Push-Relabel multiple times
    auto pr1 = engine.push_relabel(0, 3);
    auto pr2 = engine.push_relabel(0, 3);
    EXPECT_EQ(pr1.max_flow, pr2.max_flow);
    EXPECT_EQ(d1.max_flow, pr1.max_flow);

    // Alternate runs
    auto d3 = engine.dinic(0, 3);
    EXPECT_EQ(d3.max_flow, 20);
    auto pr3 = engine.push_relabel(0, 3);
    EXPECT_EQ(pr3.max_flow, 20);
}

TEST(MaxFlowTest, CyclicFlowNetwork) {
    // 0 -> 1 -> 2 -> 1 (cycle) with outgoing paths to sink 3
    MaxFlowEngine<int64_t> engine(4);
    engine.add_edge(0, 1, 15);
    engine.add_edge(1, 2, 10);
    engine.add_edge(2, 1, 5); // cycle
    engine.add_edge(2, 3, 10);
    engine.add_edge(1, 3, 5);

    auto res_dinic = engine.dinic(0, 3);
    verify_flow_invariants(4, 0, 3, res_dinic, 15);

    auto res_pr = engine.push_relabel(0, 3);
    verify_flow_invariants(4, 0, 3, res_pr, 15);
}

TEST(MaxFlowTest, DenseCompleteNetwork) {
    constexpr std::size_t n = 6;
    MaxFlowEngine<int64_t> engine(n);
    for (VertexId u = 0; u < n; ++u) {
        for (VertexId v = 0; v < n; ++v) {
            if (u != v) {
                engine.add_edge(u, v, static_cast<int64_t>((u + 1) * (v + 1)));
            }
        }
    }

    auto res_dinic = engine.dinic(0, n - 1);
    auto res_pr = engine.push_relabel(0, n - 1);

    EXPECT_EQ(res_dinic.max_flow, res_pr.max_flow);
    verify_flow_invariants(n, 0, n - 1, res_dinic, res_dinic.max_flow);
    verify_flow_invariants(n, 0, n - 1, res_pr, res_pr.max_flow);
}

TEST(MaxFlowTest, FiftyThousandRandomizedGraphs) {
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_int_distribution<std::size_t> v_dist(2, 12);
    std::uniform_int_distribution<std::size_t> e_dist(1, 24);
    std::uniform_int_distribution<int64_t> cap_dist(1, 100);

    for (std::size_t iter = 0; iter < 50000; ++iter) {
        const std::size_t num_v = v_dist(rng);
        const std::size_t num_e = e_dist(rng);

        const VertexId source = static_cast<VertexId>(rng() % num_v);
        VertexId sink = static_cast<VertexId>(rng() % num_v);
        while (sink == source) {
            sink = static_cast<VertexId>(rng() % num_v);
        }

        MaxFlowEngine<int64_t> engine(num_v);
        for (std::size_t e = 0; e < num_e; ++e) {
            const VertexId u = static_cast<VertexId>(rng() % num_v);
            const VertexId v = static_cast<VertexId>(rng() % num_v);
            const int64_t cap = cap_dist(rng);
            engine.add_edge(u, v, cap);
        }

        const auto res_dinic = engine.dinic(source, sink);
        const auto res_pr = engine.push_relabel(source, sink);

        ASSERT_EQ(res_dinic.max_flow, res_pr.max_flow);
        ASSERT_GE(res_dinic.max_flow, 0);

        // Periodically verify full invariants across all vertices
        if (iter % 100 == 0) {
            verify_flow_invariants(num_v, source, sink, res_dinic, res_dinic.max_flow);
            verify_flow_invariants(num_v, source, sink, res_pr, res_pr.max_flow);
        }
    }
}

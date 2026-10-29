#include "algoat/graph/max_flow.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <tuple>
#include <unordered_set>
#include <vector>

using namespace algoat::graph;

namespace {

void verify_result_invariants(std::size_t num_v, VertexId s, VertexId t,
                              const MaxFlowResult<int64_t>& result) {
    RC_ASSERT(result.max_flow >= 0);

    std::vector<int64_t> net_flow(num_v, 0);

    for (const auto& e : result.residual_edges) {
        RC_ASSERT(e.flow >= 0);
        RC_ASSERT(e.flow <= e.capacity);

        net_flow[e.from] -= e.flow;
        net_flow[e.to] += e.flow;
    }

    if (s != t && s < num_v && t < num_v) {
        RC_ASSERT(-net_flow[s] == result.max_flow);
        RC_ASSERT(net_flow[t] == result.max_flow);

        for (VertexId v = 0; v < num_v; ++v) {
            if (v != s && v != t) {
                RC_ASSERT(net_flow[v] == 0);
            }
        }

        std::unordered_set<VertexId> s_set(result.min_cut_source_set.begin(),
                                           result.min_cut_source_set.end());
        RC_ASSERT(s_set.contains(s));
        RC_ASSERT(!s_set.contains(t));

        int64_t cut_capacity = 0;
        for (const auto& e : result.residual_edges) {
            if (s_set.contains(e.from) && !s_set.contains(e.to)) {
                cut_capacity += e.capacity;
            }
        }
        RC_ASSERT(cut_capacity == result.max_flow);
    }
}

} // namespace

// ============================================================================
// 1. Dinic and Push-Relabel Exact Max Flow Agreement
// ============================================================================

TEST(MaxFlowPBT, DinicAndPushRelabelExactAgreement) {
    rc::check(
        "Dinic and Push-Relabel algorithms find identical maximum flow values on arbitrary graphs",
        [](std::size_t num_v,
           const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges, VertexId raw_s,
           VertexId raw_t) {
            num_v = (num_v % 32) + 2; // 2 to 33 vertices
            VertexId s = raw_s % num_v;
            VertexId t = raw_t % num_v;
            RC_PRE(s != t);

            MaxFlowEngine<int64_t> engine(num_v);
            for (const auto& [u_raw, v_raw, cap_raw] : raw_edges) {
                VertexId u = u_raw % num_v;
                VertexId v = v_raw % num_v;
                int64_t cap = static_cast<int64_t>((cap_raw % 500) + 1);
                engine.add_edge(u, v, cap);
            }

            auto res_dinic = engine.dinic(s, t);
            auto res_pr = engine.push_relabel(s, t);

            RC_ASSERT(res_dinic.max_flow == res_pr.max_flow);
            verify_result_invariants(num_v, s, t, res_dinic);
            verify_result_invariants(num_v, s, t, res_pr);
        });
}

// ============================================================================
// 2. Unit Network Invariant (Bipartite / Unit Capacities)
// ============================================================================

TEST(MaxFlowPBT, UnitNetworkInvariants) {
    rc::check("Max flow invariants hold on arbitrary unit capacity networks",
              [](std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges,
                 VertexId raw_s, VertexId raw_t) {
                  num_v = (num_v % 28) + 2;
                  VertexId s = raw_s % num_v;
                  VertexId t = raw_t % num_v;
                  RC_PRE(s != t);

                  MaxFlowEngine<int64_t> engine(num_v);
                  for (const auto& [u_raw, v_raw] : raw_edges) {
                      VertexId u = u_raw % num_v;
                      VertexId v = v_raw % num_v;
                      engine.add_edge(u, v, 1); // unit capacity
                  }

                  auto res_dinic = engine.dinic(s, t);
                  auto res_pr = engine.push_relabel(s, t);

                  RC_ASSERT(res_dinic.max_flow == res_pr.max_flow);
                  verify_result_invariants(num_v, s, t, res_dinic);
                  verify_result_invariants(num_v, s, t, res_pr);
              });
}

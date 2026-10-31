#include "algoat/graph/bipartite_matching.hpp"
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

void verify_pbt_invariants(const BipartiteGraph& graph, const BipartiteMatchingResult& matching) {
    const std::size_t num_left = graph.num_left();
    const std::size_t num_right = graph.num_right();

    RC_ASSERT(matching.match_left.size() == num_left);
    RC_ASSERT(matching.match_right.size() == num_right);

    // 1. Matching validity
    std::size_t count_left = 0;
    for (VertexId u = 0; u < num_left; ++u) {
        const VertexId v = matching.match_left[u];
        if (v != BipartiteGraph::kUnmatched) {
            RC_ASSERT(v < num_right);
            RC_ASSERT(matching.match_right[v] == u);
            const auto nbrs = graph.adj(u);
            RC_ASSERT(std::find(nbrs.begin(), nbrs.end(), v) != nbrs.end());
            count_left++;
        }
    }
    RC_ASSERT(count_left == matching.matching_size);

    std::size_t count_right = 0;
    for (VertexId v = 0; v < num_right; ++v) {
        const VertexId u = matching.match_right[v];
        if (u != BipartiteGraph::kUnmatched) {
            RC_ASSERT(u < num_left);
            RC_ASSERT(matching.match_left[u] == v);
            count_right++;
        }
    }
    RC_ASSERT(count_right == matching.matching_size);

    // 2. Minimum Vertex Cover (König's Theorem)
    const auto [cover_l, cover_r] = graph.minimum_vertex_cover(matching);

    // Duality: |Cover| == |Matching|
    RC_ASSERT(cover_l.size() + cover_r.size() == matching.matching_size);

    std::vector<bool> in_cover_l(num_left, false);
    for (VertexId u : cover_l) {
        RC_ASSERT(u < num_left);
        in_cover_l[u] = true;
    }
    std::vector<bool> in_cover_r(num_right, false);
    for (VertexId v : cover_r) {
        RC_ASSERT(v < num_right);
        in_cover_r[v] = true;
    }

    // Every edge must be covered
    for (VertexId u = 0; u < num_left; ++u) {
        for (VertexId v : graph.adj(u)) {
            RC_ASSERT(in_cover_l[u] || in_cover_r[v]);
        }
    }

    // 3. Maximum Independent Set
    const auto [indep_l, indep_r] = graph.maximum_independent_set(matching);

    // Gallai's Identity
    RC_ASSERT(indep_l.size() + indep_r.size() == num_left + num_right - matching.matching_size);

    std::vector<bool> in_indep_r(num_right, false);
    for (VertexId v : indep_r) {
        RC_ASSERT(v < num_right);
        in_indep_r[v] = true;
    }

    for (VertexId u : indep_l) {
        RC_ASSERT(u < num_left);
        for (VertexId v : graph.adj(u)) {
            RC_ASSERT(!in_indep_r[v]);
        }
    }

    // 4. Berge's Lemma Invariant
    RC_ASSERT(BipartiteMatchingTestAccess::verify_berge_invariant(graph, matching));
}

} // namespace

TEST(BipartiteMatchingPBT, KonigDualityAndGallaiInvariants) {
    rc::check("Hopcroft-Karp matching satisfies Konig duality and Gallai identity",
              [](std::size_t raw_nl, std::size_t raw_nr,
                 const std::vector<std::pair<VertexId, VertexId>>& raw_edges) {
                  const std::size_t nl = (raw_nl % 30) + 1;
                  const std::size_t nr = (raw_nr % 30) + 1;

                  BipartiteGraph graph(nl, nr);
                  for (const auto& [u_raw, v_raw] : raw_edges) {
                      const VertexId u = u_raw % nl;
                      const VertexId v = v_raw % nr;
                      graph.add_edge(u, v);
                  }

                  const auto matching = graph.maximum_matching();
                  verify_pbt_invariants(graph, matching);
              });
}

TEST(BipartiteMatchingPBT, AgreementWithDinicMaxFlowReduction) {
    rc::check(
        "Hopcroft-Karp matching matches exact Dinic maximum flow reduction",
        [](std::size_t raw_nl, std::size_t raw_nr,
           const std::vector<std::pair<VertexId, VertexId>>& raw_edges) {
            const std::size_t nl = (raw_nl % 25) + 1;
            const std::size_t nr = (raw_nr % 25) + 1;

            BipartiteGraph graph(nl, nr);

            // Setup unit flow network: source s = nl + nr, sink t = nl + nr + 1
            const VertexId s = static_cast<VertexId>(nl + nr);
            const VertexId t = static_cast<VertexId>(nl + nr + 1);
            MaxFlowEngine<int64_t> flow_engine(nl + nr + 2);

            for (VertexId u = 0; u < nl; ++u) {
                flow_engine.add_edge(s, u, 1);
            }
            for (VertexId v = 0; v < nr; ++v) {
                flow_engine.add_edge(static_cast<VertexId>(nl + v), t, 1);
            }

            std::unordered_set<uint64_t> added_edges;
            for (const auto& [u_raw, v_raw] : raw_edges) {
                const VertexId u = u_raw % nl;
                const VertexId v = v_raw % nr;
                const uint64_t key = (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
                if (added_edges.insert(key).second) {
                    graph.add_edge(u, v);
                    flow_engine.add_edge(u, static_cast<VertexId>(nl + v), 1);
                }
            }

            const auto hk_matching = graph.maximum_matching();
            const auto dinic_result = flow_engine.dinic(s, t);

            RC_ASSERT(hk_matching.matching_size == static_cast<std::size_t>(dinic_result.max_flow));
            verify_pbt_invariants(graph, hk_matching);
        });
}

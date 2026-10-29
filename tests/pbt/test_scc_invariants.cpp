#include "../graph/scc_test_utils.hpp"
#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/scc.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <utility>
#include <vector>

using namespace algoat::graph;
using algoat::graph::test::KosarajuReference;

namespace {

std::vector<std::pair<VertexId, VertexId>>
generate_random_edges(std::size_t num_v,
                      const std::vector<std::pair<uint32_t, uint32_t>>& raw_edges) {
    std::vector<std::pair<VertexId, VertexId>> valid_edges;
    valid_edges.reserve(raw_edges.size());
    for (const auto& [raw_u, raw_v] : raw_edges) {
        valid_edges.emplace_back(raw_u % num_v, raw_v % num_v);
    }
    return valid_edges;
}

} // namespace

// ============================================================================
// 1. RapidCheck: Partition Invariants & Completeness
// ============================================================================

TEST(SCCPBT, PartitionCompletenessAndInvariants) {
    rc::check("SCC partition covers all vertices and component lists are disjoint and complete",
              [](std::size_t num_v, const std::vector<std::pair<uint32_t, uint32_t>>& raw_edges) {
                  num_v = (num_v % 64) + 1; // 1 to 64 vertices
                  const auto edges = generate_random_edges(num_v, raw_edges);
                  auto g = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/true);
                  StronglyConnectedComponents engine(g);

                  auto res = engine.compute();

                  RC_ASSERT(res.num_components > 0);
                  RC_ASSERT(res.num_components <= num_v);
                  RC_ASSERT(res.component_ids.size() == num_v);
                  RC_ASSERT(res.components.size() == res.num_components);

                  std::size_t total_members = 0;
                  for (uint32_t c = 0; c < res.num_components; ++c) {
                      RC_ASSERT(!res.components[c].empty());
                      total_members += res.components[c].size();
                      for (VertexId u : res.components[c]) {
                          RC_ASSERT(res.component_ids[u] == c);
                      }
                  }
                  RC_ASSERT(total_members == num_v);
              });
}

// ============================================================================
// 2. RapidCheck: Condensation Graph DAG & Topological Order Invariants
// ============================================================================

TEST(SCCPBT, CondensationGraphIsDAGAndTopologicallySorted) {
    rc::check("Condensation graph is strictly acyclic, has no self-loops, and satisfies A < B",
              [](std::size_t num_v, const std::vector<std::pair<uint32_t, uint32_t>>& raw_edges) {
                  num_v = (num_v % 64) + 1;
                  const auto edges = generate_random_edges(num_v, raw_edges);
                  auto g = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/true);
                  StronglyConnectedComponents engine(g);

                  auto res = engine.compute();
                  auto dag = engine.condensation_graph(res);

                  RC_ASSERT(dag.num_vertices() == res.num_components);
                  RC_ASSERT(dag.is_directed());

                  for (VertexId u = 0; u < dag.num_vertices(); ++u) {
                      for (VertexId v : dag.out_neighbors(u)) {
                          // Invariant: strictly upper-triangular edge orientation
                          RC_ASSERT(u < v);
                      }
                  }
              });
}

// ============================================================================
// 3. RapidCheck: Equivalence Against Independent Kosaraju Reference
// ============================================================================

TEST(SCCPBT, ExactEquivalenceAgainstKosarajuReference) {
    rc::check("SCC partition is identical to independent Kosaraju algorithm across random graphs",
              [](std::size_t num_v, const std::vector<std::pair<uint32_t, uint32_t>>& raw_edges) {
                  num_v = (num_v % 50) + 1;
                  const auto edges = generate_random_edges(num_v, raw_edges);
                  auto g = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/true);
                  StronglyConnectedComponents engine(g);
                  auto res = engine.compute();

                  uint32_t ref_num_scc = 0;
                  auto ref_comp = KosarajuReference::compute(num_v, edges, ref_num_scc);

                  RC_ASSERT(res.num_components == ref_num_scc);

                  // Assert that two vertices are in the same component iff they are in the same
                  // component under Kosaraju
                  for (VertexId u = 0; u < num_v; ++u) {
                      for (VertexId v = u + 1; v < num_v; ++v) {
                          bool same_in_engine = (res.component_ids[u] == res.component_ids[v]);
                          bool same_in_ref = (ref_comp[u] == ref_comp[v]);
                          RC_ASSERT(same_in_engine == same_in_ref);
                      }
                  }
              });
}

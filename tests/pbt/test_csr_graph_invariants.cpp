#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <utility>
#include <vector>

using namespace algoat::graph;

namespace {

std::vector<std::pair<VertexId, VertexId>>
filter_valid_edges(std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges) {
    std::vector<std::pair<VertexId, VertexId>> valid_edges;
    valid_edges.reserve(raw_edges.size());
    for (const auto& [u, v] : raw_edges) {
        valid_edges.emplace_back(u % num_v, v % num_v);
    }
    return valid_edges;
}

} // namespace

// ============================================================================
// 1. Degree Sum Equals Total Edges Invariant
// ============================================================================

TEST(CSRGraphPBT, DegreeSumMatchesTotalEdges) {
    rc::check("Sum of out_degrees matches num_edges for both directed and undirected",
              [](std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges,
                 bool directed) {
                  num_v = (num_v % 500) + 1; // 1 to 500 vertices
                  const auto valid_edges = filter_valid_edges(num_v, raw_edges);
                  CSRGraph g = CSRGraph<>::build_from_edges(num_v, valid_edges, directed);

                  EdgeIndex out_deg_sum = 0;
                  EdgeIndex in_deg_sum = 0;
                  for (VertexId u = 0; u < g.num_vertices(); ++u) {
                      out_deg_sum += g.out_degree(u);
                      in_deg_sum += g.in_degree(u);
                  }

                  RC_ASSERT(out_deg_sum == g.num_edges());
                  RC_ASSERT(in_deg_sum == g.num_edges());
              });
}

// ============================================================================
// 2. Bidirectional Edge Duality Invariant
// ============================================================================

TEST(CSRGraphPBT, BidirectionalEdgeDuality) {
    rc::check("v in out_neighbors(u) if and only if u in in_neighbors(v)",
              [](std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges,
                 bool directed) {
                  num_v = (num_v % 300) + 1;
                  const auto valid_edges = filter_valid_edges(num_v, raw_edges);
                  CSRGraph g = CSRGraph<>::build_from_edges(num_v, valid_edges, directed);

                  for (VertexId u = 0; u < g.num_vertices(); ++u) {
                      for (VertexId v : g.out_neighbors(u)) {
                          auto in_v = g.in_neighbors(v);
                          bool found = std::binary_search(in_v.begin(), in_v.end(), u);
                          RC_ASSERT(found);
                      }
                      for (VertexId v : g.in_neighbors(u)) {
                          auto out_v = g.out_neighbors(v);
                          bool found = std::binary_search(out_v.begin(), out_v.end(), u);
                          RC_ASSERT(found);
                      }
                  }
              });
}

// ============================================================================
// 3. Double Transposition Involution Invariant
// ============================================================================

TEST(CSRGraphPBT, DoubleTransposeInvolution) {
    rc::check("g.transpose().transpose() preserves identical topology",
              [](std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges) {
                  num_v = (num_v % 300) + 1;
                  const auto valid_edges = filter_valid_edges(num_v, raw_edges);
                  CSRGraph g = CSRGraph<>::build_from_edges(num_v, valid_edges, /*directed=*/true);
                  CSRGraph g_double_t = g.transpose().transpose();

                  RC_ASSERT(g_double_t.num_vertices() == g.num_vertices());
                  RC_ASSERT(g_double_t.num_edges() == g.num_edges());

                  for (VertexId u = 0; u < g.num_vertices(); ++u) {
                      auto orig_out = g.out_neighbors(u);
                      auto t_out = g_double_t.out_neighbors(u);
                      RC_ASSERT(orig_out.size() == t_out.size());
                      for (std::size_t i = 0; i < orig_out.size(); ++i) {
                          RC_ASSERT(orig_out[i] == t_out[i]);
                      }

                      auto orig_in = g.in_neighbors(u);
                      auto t_in = g_double_t.in_neighbors(u);
                      RC_ASSERT(orig_in.size() == t_in.size());
                      for (std::size_t i = 0; i < orig_in.size(); ++i) {
                          RC_ASSERT(orig_in[i] == t_in[i]);
                      }
                  }
              });
}

// ============================================================================
// 4. Neighbors Strictly Sorted and Unique Invariant
// ============================================================================

TEST(CSRGraphPBT, NeighborsStrictlySortedAndUnique) {
    rc::check("All out_neighbors and in_neighbors are strictly sorted with zero duplicates",
              [](std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges,
                 bool directed) {
                  num_v = (num_v % 400) + 1;
                  const auto valid_edges = filter_valid_edges(num_v, raw_edges);
                  CSRGraph g = CSRGraph<>::build_from_edges(num_v, valid_edges, directed);

                  for (VertexId u = 0; u < g.num_vertices(); ++u) {
                      auto out_neigh = g.out_neighbors(u);
                      RC_ASSERT(std::is_sorted(out_neigh.begin(), out_neigh.end()));
                      RC_ASSERT(std::adjacent_find(out_neigh.begin(), out_neigh.end()) ==
                                out_neigh.end());

                      auto in_neigh = g.in_neighbors(u);
                      RC_ASSERT(std::is_sorted(in_neigh.begin(), in_neigh.end()));
                      RC_ASSERT(std::adjacent_find(in_neigh.begin(), in_neigh.end()) ==
                                in_neigh.end());
                  }
              });
}

// ============================================================================
// 5. Undirected Memory Footprint Bound Invariant
// ============================================================================

TEST(CSRGraphPBT, UndirectedMemoryBoundStrictlySatisfied) {
    rc::check("Undirected memory footprint strictly equals (V + 1) * 8 + E * 4 bytes",
              [](std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges) {
                  num_v = (num_v % 500) + 1;
                  const auto valid_edges = filter_valid_edges(num_v, raw_edges);
                  CSRGraph g = CSRGraph<>::build_from_edges(num_v, valid_edges, /*directed=*/false);

                  std::size_t expected_bytes =
                      (num_v + 1) * sizeof(EdgeIndex) + g.num_edges() * sizeof(VertexId);
                  RC_ASSERT(g.memory_footprint() == expected_bytes);
              });
}

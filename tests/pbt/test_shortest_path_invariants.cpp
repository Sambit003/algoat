#include "algoat/graph/shortest_path.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <tuple>
#include <utility>
#include <vector>

using namespace algoat::graph;

namespace {

std::vector<std::tuple<VertexId, VertexId, double>> generate_valid_weighted_edges(
    std::size_t num_v, const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges) {
    std::vector<std::tuple<VertexId, VertexId, double>> valid_edges;
    valid_edges.reserve(raw_edges.size());
    for (const auto& [u, v, raw_w] : raw_edges) {
        double w = static_cast<double>((raw_w % 1000) + 1); // strictly positive weight [1, 1000]
        valid_edges.emplace_back(u % num_v, v % num_v, w);
    }
    return valid_edges;
}

} // namespace

// ============================================================================
// 1. Dijkstra and ALT Distance Exact Agreement
// ============================================================================

TEST(ShortestPathPBT, DijkstraALTDistanceExactAgreement) {
    rc::check("ALT search distance matches Dijkstra distance exactly on arbitrary valid graphs",
              [](std::size_t num_v,
                 const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges,
                 bool directed, VertexId raw_s, VertexId raw_t) {
                  num_v = (num_v % 60) + 2; // 2 to 61 vertices
                  const auto edges = generate_valid_weighted_edges(num_v, raw_edges);
                  auto g = CSRGraph<double>::build_from_weighted_edges(num_v, edges, directed);
                  ShortestPathEngine<double> engine(g);
                  engine.preprocess_landmarks(std::min<std::size_t>(6, num_v));

                  VertexId s = raw_s % num_v;
                  VertexId t = raw_t % num_v;

                  std::vector<VertexId> path_dijkstra;
                  auto dist_d = engine.dijkstra_p2p(s, t, &path_dijkstra);

                  std::vector<VertexId> path_alt;
                  auto dist_alt = engine.alt_search(s, t, &path_alt);

                  RC_ASSERT(dist_d.has_value() == dist_alt.has_value());
                  if (dist_d.has_value()) {
                      RC_ASSERT(std::abs(*dist_d - *dist_alt) < 1e-6);
                      if (!path_alt.empty()) {
                          RC_ASSERT(path_alt.front() == s);
                          RC_ASSERT(path_alt.back() == t);
                      }
                  }
              });
}

// ============================================================================
// 2. Triangle Inequality Invariant
// ============================================================================

TEST(ShortestPathPBT, MetricTriangleInequality) {
    rc::check("Shortest path distance satisfies triangle inequality: d(u, w) <= d(u, v) + d(v, w)",
              [](std::size_t num_v,
                 const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges,
                 VertexId raw_u, VertexId raw_v, VertexId raw_w) {
                  num_v = (num_v % 50) + 3; // 3 to 52 vertices
                  const auto edges = generate_valid_weighted_edges(num_v, raw_edges);
                  auto g =
                      CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/true);
                  ShortestPathEngine<double> engine(g);

                  VertexId u = raw_u % num_v;
                  VertexId v = raw_v % num_v;
                  VertexId w = raw_w % num_v;

                  auto d_uv = engine.dijkstra_p2p(u, v);
                  auto d_vw = engine.dijkstra_p2p(v, w);
                  auto d_uw = engine.dijkstra_p2p(u, w);

                  if (d_uv.has_value() && d_vw.has_value()) {
                      RC_ASSERT(d_uw.has_value());
                      RC_ASSERT(*d_uw <= *d_uv + *d_vw + 1e-6);
                  }
              });
}

// ============================================================================
// 3. Bidirectional Search Distance Invariant
// ============================================================================

TEST(ShortestPathPBT, BidirectionalSearchMatchesSSSP) {
    rc::check("Bidirectional Dijkstra and ALT match unidirectional Dijkstra distance",
              [](std::size_t num_v,
                 const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges,
                 VertexId raw_s, VertexId raw_t) {
                  num_v = (num_v % 50) + 2;
                  const auto edges = generate_valid_weighted_edges(num_v, raw_edges);
                  auto g =
                      CSRGraph<double>::build_from_weighted_edges(num_v, edges, /*directed=*/false);
                  ShortestPathEngine<double> engine(g);
                  engine.preprocess_landmarks(std::min<std::size_t>(6, num_v));

                  VertexId s = raw_s % num_v;
                  VertexId t = raw_t % num_v;

                  auto dist_d = engine.dijkstra_p2p(s, t);
                  auto dist_bi = engine.dijkstra_bidirectional(s, t);
                  auto dist_alt_bi = engine.alt_search_bidirectional(s, t);

                  RC_ASSERT(dist_d.has_value() == dist_bi.has_value());
                  RC_ASSERT(dist_d.has_value() == dist_alt_bi.has_value());
                  if (dist_d.has_value()) {
                      RC_ASSERT(std::abs(*dist_d - *dist_bi) < 1e-6);
                      RC_ASSERT(std::abs(*dist_d - *dist_alt_bi) < 1e-6);
                  }
              });
}

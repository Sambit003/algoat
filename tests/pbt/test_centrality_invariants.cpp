#include "algoat/graph/centrality.hpp"
#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>
#include <rapidcheck.h>
#include <tuple>
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

std::vector<std::tuple<VertexId, VertexId, double>> filter_valid_weighted_edges(
    std::size_t num_v, const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges) {
    std::vector<std::tuple<VertexId, VertexId, double>> valid_edges;
    valid_edges.reserve(raw_edges.size());
    for (const auto& [u, v, raw_w] : raw_edges) {
        double w = static_cast<double>((raw_w % 500) + 1); // strictly positive weight [1, 500]
        valid_edges.emplace_back(u % num_v, v % num_v, w);
    }
    return valid_edges;
}

} // namespace

// ============================================================================
// 1. PageRank Total Probability Conservation Invariant (sum == 1.0)
// ============================================================================

TEST(CentralityPBT, PageRankProbabilityConservation) {
    rc::check("PageRank scores strictly sum to 1.0 +- 1e-9 on any valid graph",
              [](std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges,
                 bool directed) {
                  num_v = (num_v % 64) + 1; // 1 to 64 vertices
                  const auto valid_edges = filter_valid_edges(num_v, raw_edges);
                  auto g = CSRGraph<void>::build_from_edges(num_v, valid_edges, directed);

                  PageRankConfig config;
                  config.tolerance = 1e-7;
                  config.max_iterations = 80;
                  auto pr = pagerank(g, config);

                  RC_ASSERT(pr.scores.size() == num_v);
                  double sum = std::accumulate(pr.scores.begin(), pr.scores.end(), 0.0);
                  RC_ASSERT(std::abs(sum - 1.0) < 1e-9);

                  for (double score : pr.scores) {
                      RC_ASSERT(score >= 0.0);
                      RC_ASSERT(score <= 1.0 + 1e-9);
                  }
              });
}

// ============================================================================
// 2. Betweenness & Degree Centrality Non-Negativity & Bounds Invariant
// ============================================================================

TEST(CentralityPBT, BetweennessAndDegreeCentralityBounds) {
    rc::check("Betweenness is non-negative and degree centrality is bounded",
              [](std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges,
                 bool directed) {
                  num_v = (num_v % 40) + 1; // 1 to 40 vertices
                  const auto valid_edges = filter_valid_edges(num_v, raw_edges);
                  auto g = CSRGraph<void>::build_from_edges(num_v, valid_edges, directed);

                  auto result = betweenness_centrality(g);
                  RC_ASSERT(result.betweenness.size() == num_v);
                  RC_ASSERT(result.degree_centrality.size() == num_v);

                  for (std::size_t i = 0; i < num_v; ++i) {
                      RC_ASSERT(result.betweenness[i] >= 0.0);
                      RC_ASSERT(!std::isnan(result.betweenness[i]));
                      RC_ASSERT(!std::isinf(result.betweenness[i]));

                      RC_ASSERT(result.degree_centrality[i] >= 0.0);
                      // In directed graph, max degree centrality can be up to 2.0; undirected
                      // <= 1.0
                      const double max_allowed = directed ? 2.0 + 1e-9 : 1.0 + 1e-9;
                      RC_ASSERT(result.degree_centrality[i] <= max_allowed);
                  }
              });
}

// ============================================================================
// 3. Isolated Vertices Have Zero Betweenness
// ============================================================================

TEST(CentralityPBT, IsolatedVerticesZeroBetweenness) {
    rc::check("Vertices with degree 0 strictly have betweenness 0.0",
              [](std::size_t num_v, const std::vector<std::pair<VertexId, VertexId>>& raw_edges,
                 bool directed) {
                  num_v = (num_v % 30) + 2; // 2 to 31 vertices
                  const auto valid_edges = filter_valid_edges(num_v, raw_edges);
                  auto g = CSRGraph<void>::build_from_edges(num_v, valid_edges, directed);

                  auto result = betweenness_centrality(g);
                  for (VertexId u = 0; u < num_v; ++u) {
                      if (g.out_degree(u) == 0 && g.in_degree(u) == 0) {
                          RC_ASSERT(result.betweenness[u] == 0.0);
                          RC_ASSERT(result.degree_centrality[u] == 0.0);
                      }
                  }
              });
}

// ============================================================================
// 4. Weighted Betweenness Centrality Non-Negativity
// ============================================================================

TEST(CentralityPBT, WeightedBetweennessNonNegativity) {
    rc::check(
        "Weighted graph betweenness scores are strictly non-negative",
        [](std::size_t num_v,
           const std::vector<std::tuple<VertexId, VertexId, uint32_t>>& raw_edges, bool directed) {
            num_v = (num_v % 30) + 1; // 1 to 30 vertices
            const auto valid_edges = filter_valid_weighted_edges(num_v, raw_edges);
            auto g = CSRGraph<double>::build_from_weighted_edges(num_v, valid_edges, directed);

            auto result = betweenness_centrality(g);
            RC_ASSERT(result.betweenness.size() == num_v);
            for (double bc : result.betweenness) {
                RC_ASSERT(bc >= 0.0);
                RC_ASSERT(!std::isnan(bc));
                RC_ASSERT(!std::isinf(bc));
            }
        });
}

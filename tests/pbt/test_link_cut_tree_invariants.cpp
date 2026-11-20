#include "../graph/link_cut_tree_test_helpers.hpp"
#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/disjoint_set.hpp"
#include "algoat/graph/link_cut_tree.hpp"
#include "algoat/graph/mst.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <rapidcheck.h>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using algoat::graph::CSRGraph;
using algoat::graph::DisjointSetUnion;
using algoat::graph::LinkCutTree;
using algoat::graph::MinimumSpanningTreeEngine;
using algoat::graph::VertexId;
using algoat::graph::testing::DynamicMstMaintainer;
using algoat::graph::testing::ForestOracle;

TEST(LinkCutTreePBT, DifferentialRandomOperationsMatchOracle) {
    rc::check("LinkCutTree matches independent forest oracle across arbitrary operation sequences",
              [](std::size_t raw_n, const std::vector<int32_t>& init_vals,
                 const std::vector<std::tuple<uint8_t, uint8_t, uint8_t, int32_t>>& ops) {
                  const std::size_t n = (raw_n % 28) + 1;
                  LinkCutTree<int64_t> lct(n);
                  ForestOracle<int64_t> oracle(n);
                  std::vector<std::pair<std::size_t, std::size_t>> edges;

                  for (std::size_t i = 0; i < n; ++i) {
                      const int64_t val =
                          i < init_vals.size() ? static_cast<int64_t>(init_vals[i]) : int64_t{0};
                      lct.update_vertex(i, val);
                      oracle.update_vertex(i, val);
                  }

                  const std::size_t step_limit = std::min<std::size_t>(ops.size(), 250);
                  for (std::size_t i = 0; i < step_limit; ++i) {
                      const auto [op_raw, u_raw, v_raw, val_raw] = ops[i];
                      const std::size_t op = op_raw % 8;
                      const std::size_t u = u_raw % n;
                      const std::size_t v = v_raw % n;

                      switch (op) {
                      case 0: // link
                          if (u != v && !oracle.connected(u, v)) {
                              if (!oracle.is_root(u)) {
                                  RC_ASSERT_THROWS_AS(lct.link(u, v), std::invalid_argument);
                                  lct.make_root(u);
                                  oracle.make_root(u);
                              }
                              lct.link(u, v);
                              oracle.link(u, v);
                              edges.emplace_back(u, v);
                          } else {
                              RC_ASSERT_THROWS_AS(lct.link(u, v), std::invalid_argument);
                          }
                          break;
                      case 1: // cut
                          if (!edges.empty() && (op_raw & 1U) == 0U) {
                              const std::size_t idx =
                                  (static_cast<std::size_t>(u_raw) + v_raw) % edges.size();
                              auto [eu, ev] = edges[idx];
                              if ((v_raw & 1U) != 0U) {
                                  std::swap(eu, ev);
                              }
                              lct.cut(eu, ev);
                              oracle.cut(eu, ev);
                              edges[idx] = edges.back();
                              edges.pop_back();
                          } else if (u != v && oracle.has_edge(u, v)) {
                              lct.cut(u, v);
                              oracle.cut(u, v);
                              for (std::size_t idx = 0; idx < edges.size(); ++idx) {
                                  if ((edges[idx].first == u && edges[idx].second == v) ||
                                      (edges[idx].first == v && edges[idx].second == u)) {
                                      edges[idx] = edges.back();
                                      edges.pop_back();
                                      break;
                                  }
                              }
                          } else {
                              RC_ASSERT_THROWS_AS(lct.cut(u, v), std::invalid_argument);
                          }
                          break;
                      case 2: // make_root
                          lct.make_root(u);
                          oracle.make_root(u);
                          break;
                      case 3: // update_vertex
                          lct.update_vertex(u, static_cast<int64_t>(val_raw));
                          oracle.update_vertex(u, static_cast<int64_t>(val_raw));
                          break;
                      case 4: // connected
                          RC_ASSERT(lct.connected(u, v) == oracle.connected(u, v));
                          break;
                      case 5: // find_root
                          RC_ASSERT(lct.find_root(u) == oracle.find_root(u));
                          break;
                      case 6: // query_path_sum
                          if (oracle.connected(u, v)) {
                              const auto [exp_sum, exp_min] = oracle.query_path(u, v);
                              (void)exp_min;
                              RC_ASSERT(lct.query_path_sum(u, v) == exp_sum);
                          } else {
                              RC_ASSERT_THROWS_AS(static_cast<void>(lct.query_path_sum(u, v)),
                                                  std::invalid_argument);
                          }
                          break;
                      case 7: // query_path_min
                          if (oracle.connected(u, v)) {
                              const auto [exp_sum, exp_min] = oracle.query_path(u, v);
                              (void)exp_sum;
                              RC_ASSERT(lct.query_path_min(u, v) == exp_min);
                          } else {
                              RC_ASSERT_THROWS_AS(static_cast<void>(lct.query_path_min(u, v)),
                                                  std::invalid_argument);
                          }
                          break;
                      default:
                          break;
                      }
                  }

                  for (std::size_t u = 0; u < n; ++u) {
                      RC_ASSERT(lct.find_root(u) == oracle.find_root(u));
                  }
              });
}

TEST(LinkCutTreePBT, AlgebraicSymmetryAndRootPreservation) {
    rc::check(
        "Read queries, failed mutations, and updates preserve tree roots and path symmetry",
        [](std::size_t raw_n, const std::vector<std::tuple<uint8_t, uint8_t, int16_t>>& raw_ops) {
            const std::size_t n = (raw_n % 24) + 2;
            LinkCutTree<int64_t> lct(n);
            ForestOracle<int64_t> oracle(n);

            for (const auto& [u_raw, v_raw, w] : raw_ops) {
                const std::size_t u = u_raw % n;
                const std::size_t v = v_raw % n;
                lct.update_vertex(u, static_cast<int64_t>(w));
                oracle.update_vertex(u, static_cast<int64_t>(w));
                if (u != v && !oracle.connected(u, v) && oracle.is_root(u)) {
                    lct.link(u, v);
                    oracle.link(u, v);
                }
            }

            const std::vector<std::size_t> roots_before = oracle.all_roots();
            for (std::size_t u = 0; u < n; ++u) {
                const std::size_t v = (u * 7 + 3) % n;
                RC_ASSERT(lct.connected(u, v) == lct.connected(v, u));
                if (oracle.connected(u, v)) {
                    RC_ASSERT(lct.query_path_sum(u, v) == lct.query_path_sum(v, u));
                    RC_ASSERT(lct.query_path_min(u, v) == lct.query_path_min(v, u));
                    RC_ASSERT_THROWS_AS(lct.link(u, v), std::invalid_argument);
                } else {
                    RC_ASSERT_THROWS_AS(lct.cut(u, v), std::invalid_argument);
                    RC_ASSERT_THROWS_AS(static_cast<void>(lct.query_path_sum(u, v)),
                                        std::invalid_argument);
                }
            }
            for (std::size_t u = 0; u < n; ++u) {
                RC_ASSERT(lct.find_root(u) == roots_before[u]);
            }

            // make_root(pivot) only updates roots inside pivot's component
            const std::size_t pivot = raw_n % n;
            lct.make_root(pivot);
            for (std::size_t u = 0; u < n; ++u) {
                if (oracle.connected(u, pivot)) {
                    RC_ASSERT(lct.find_root(u) == pivot);
                } else {
                    RC_ASSERT(lct.find_root(u) == roots_before[u]);
                }
            }
        });
}

TEST(LinkCutTreePBT, DynamicMstMatchesStaticKruskal) {
    rc::check(
        "Incremental Dynamic MST using LinkCutTree matches Kruskal total weight and components",
        [](std::size_t raw_n, const std::vector<std::tuple<uint8_t, uint8_t, int16_t>>& raw_edges) {
            const std::size_t n = (raw_n % 30) + 1;
            DynamicMstMaintainer dyn_mst(n);
            std::vector<std::tuple<VertexId, VertexId, int64_t>> all_edges;

            const std::size_t edge_limit = std::min<std::size_t>(raw_edges.size(), 90);
            for (std::size_t i = 0; i < edge_limit; ++i) {
                const auto [u_raw, v_raw, w_raw] = raw_edges[i];
                const std::size_t u = u_raw % n;
                const std::size_t v = v_raw % n;
                const int64_t w = static_cast<int64_t>(w_raw);
                all_edges.emplace_back(static_cast<VertexId>(u), static_cast<VertexId>(v), w);
                dyn_mst.insert_edge(u, v, w);
            }

            std::sort(all_edges.begin(), all_edges.end(),
                      [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
            DisjointSetUnion dsu(n);
            int64_t kruskal_weight = 0;
            for (const auto& [eu, ev, ew] : all_edges) {
                if (eu != ev && dsu.unite(eu, ev)) {
                    kruskal_weight += ew;
                }
            }
            RC_ASSERT(dyn_mst.total_weight() == kruskal_weight);
            RC_ASSERT(dyn_mst.components_count() == dsu.num_components());

            // Cross-check deduplicated min-weight simple graph against MinimumSpanningTreeEngine
            std::map<std::pair<VertexId, VertexId>, int64_t> min_simple;
            for (const auto& [eu, ev, ew] : all_edges) {
                if (eu == ev) {
                    continue;
                }
                const auto key = std::minmax(eu, ev);
                auto [it, inserted] = min_simple.emplace(key, ew);
                if (!inserted) {
                    it->second = std::min(it->second, ew);
                }
            }
            std::vector<std::tuple<VertexId, VertexId, int64_t>> simple_edges;
            simple_edges.reserve(min_simple.size());
            for (const auto& [uv, ew] : min_simple) {
                simple_edges.emplace_back(uv.first, uv.second, ew);
            }
            auto csr =
                CSRGraph<int64_t>::build_from_weighted_edges(n, simple_edges, /*directed=*/false);
            const auto engine_res = MinimumSpanningTreeEngine<int64_t>(csr).compute_serial();
            RC_ASSERT(dyn_mst.total_weight() == engine_res.total_weight);
            RC_ASSERT(dyn_mst.components_count() == engine_res.components_count);
        });
}

TEST(LinkCutTreePBT, FloatingPointInstantiationMatchesOracle) {
    rc::check("LinkCutTree<double> matches forest oracle path sum and min within floating-point "
              "tolerance",
              [](std::size_t raw_n, const std::vector<int16_t>& init_vals,
                 const std::vector<std::tuple<uint8_t, uint8_t, uint8_t, int16_t>>& ops) {
                  const std::size_t n = (raw_n % 20) + 1;
                  LinkCutTree<double> lct(n);
                  ForestOracle<double> oracle(n);

                  for (std::size_t i = 0; i < n; ++i) {
                      const double val =
                          i < init_vals.size() ? static_cast<double>(init_vals[i]) / 16.0 : 0.0;
                      lct.update_vertex(i, val);
                      oracle.update_vertex(i, val);
                  }

                  const std::size_t limit = std::min<std::size_t>(ops.size(), 120);
                  for (std::size_t i = 0; i < limit; ++i) {
                      const auto [op_raw, u_raw, v_raw, val_raw] = ops[i];
                      const std::size_t u = u_raw % n;
                      const std::size_t v = v_raw % n;
                      switch (op_raw % 5) {
                      case 0:
                          if (u != v && !oracle.connected(u, v)) {
                              if (!oracle.is_root(u)) {
                                  lct.make_root(u);
                                  oracle.make_root(u);
                              }
                              lct.link(u, v);
                              oracle.link(u, v);
                          }
                          break;
                      case 1:
                          if (u != v && oracle.has_edge(u, v)) {
                              lct.cut(u, v);
                              oracle.cut(u, v);
                          }
                          break;
                      case 2:
                          lct.make_root(u);
                          oracle.make_root(u);
                          break;
                      case 3: {
                          const double val = static_cast<double>(val_raw) / 16.0;
                          lct.update_vertex(u, val);
                          oracle.update_vertex(u, val);
                          break;
                      }
                      case 4:
                          if (oracle.connected(u, v)) {
                              const auto [exp_sum, exp_min] = oracle.query_path(u, v);
                              RC_ASSERT(std::abs(lct.query_path_sum(u, v) - exp_sum) < 1e-9);
                              RC_ASSERT(lct.query_path_min(u, v) == exp_min);
                          }
                          break;
                      default:
                          break;
                      }
                  }
              });
}

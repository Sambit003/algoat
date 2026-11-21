#include "../graph/biconnectivity_test_utils.hpp"
#include "algoat/graph/biconnectivity.hpp"
#include "algoat/graph/csr_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using algoat::graph::BiconnectivityResult;
using algoat::graph::BlockCutTree;
using algoat::graph::BridgeBlockTree;
using algoat::graph::build_block_cut_tree;
using algoat::graph::build_bridge_block_tree;
using algoat::graph::CSRGraph;
using algoat::graph::find_biconnectivity;
using algoat::graph::VertexId;
using algoat::graph::test::bridge_tree_vertex_groups;
using algoat::graph::test::BruteForceOracle;
using algoat::graph::test::canonicalize_components;
using algoat::graph::test::NetworkXReference;
using algoat::graph::test::verify_all_biconnectivity_invariants;

namespace {

std::vector<std::pair<VertexId, VertexId>>
normalize_edges(std::size_t num_v, const std::vector<std::pair<uint32_t, uint32_t>>& raw_edges) {
    std::vector<std::pair<VertexId, VertexId>> edges;
    if (num_v == 0) {
        return edges;
    }
    edges.reserve(raw_edges.size());
    for (const auto& [u, v] : raw_edges) {
        edges.emplace_back(static_cast<VertexId>(u % num_v), static_cast<VertexId>(v % num_v));
    }
    return edges;
}

} // namespace

TEST(BiconnectivityPBT, ExactParityWithBruteForceAndNetworkXOracles) {
    rc::check("find_biconnectivity and build_bridge_block_tree match BruteForceOracle and "
              "NetworkXReference",
              [](std::size_t raw_v, const std::vector<std::pair<uint32_t, uint32_t>>& raw_edges) {
                  const std::size_t num_v = (raw_v % 24U) + 1U;
                  const auto edges = normalize_edges(num_v, raw_edges);
                  const auto g = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/false);

                  const BiconnectivityResult res = find_biconnectivity(g);
                  const BridgeBlockTree bbt = build_bridge_block_tree(g);
                  const BlockCutTree bct = build_block_cut_tree(g);

                  RC_ASSERT(verify_all_biconnectivity_invariants(g, res, bbt, bct));
                  RC_ASSERT(res.articulation_points == BruteForceOracle::articulation_points(g));
                  RC_ASSERT(res.bridges == BruteForceOracle::bridges(g));
                  RC_ASSERT(bridge_tree_vertex_groups(bbt) ==
                            BruteForceOracle::bridge_components(g));

                  const auto nx_ref = NetworkXReference::compute(g);
                  RC_ASSERT(res.articulation_points == nx_ref.articulation_points);
                  RC_ASSERT(res.bridges == nx_ref.bridges);
                  RC_ASSERT(canonicalize_components(res.biconnected_components) ==
                            nx_ref.biconnected_components);
                  RC_ASSERT(bridge_tree_vertex_groups(bbt) == nx_ref.bridge_components);
              });
}

TEST(BiconnectivityPBT, StructuralInvariantsHoldAcrossDisconnectedAndSelfLoopGraphs) {
    rc::check("All block, bridge-block forest, and bipartite block-cut invariants hold for V in "
              "[0, 64]",
              [](std::size_t raw_v, const std::vector<std::pair<uint32_t, uint32_t>>& raw_edges) {
                  const std::size_t num_v = raw_v % 65U;
                  const auto edges = normalize_edges(num_v, raw_edges);
                  const auto g = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/false);

                  const BiconnectivityResult res = find_biconnectivity(g);
                  const BridgeBlockTree bbt = build_bridge_block_tree(g);
                  const BlockCutTree bct = build_block_cut_tree(g);

                  RC_ASSERT(verify_all_biconnectivity_invariants(g, res, bbt, bct));

                  const auto nx_ref = NetworkXReference::compute(g);
                  RC_ASSERT(res.articulation_points == nx_ref.articulation_points);
                  RC_ASSERT(res.bridges == nx_ref.bridges);
                  RC_ASSERT(canonicalize_components(res.biconnected_components) ==
                            nx_ref.biconnected_components);
                  RC_ASSERT(bridge_tree_vertex_groups(bbt) == nx_ref.bridge_components);
              });
}

TEST(BiconnectivityPBT, WeightedGraphMatchesUnweightedTopology) {
    rc::check(
        "Weighted CSRGraph<double> yields identical biconnectivity and tree condensations",
        [](std::size_t raw_v, const std::vector<std::pair<uint32_t, uint32_t>>& raw_edges) {
            const std::size_t num_v = raw_v % 48U;
            const auto edges = normalize_edges(num_v, raw_edges);
            std::vector<std::tuple<VertexId, VertexId, double>> w_edges;
            w_edges.reserve(edges.size());
            for (std::size_t i = 0; i < edges.size(); ++i) {
                w_edges.emplace_back(edges[i].first, edges[i].second,
                                     static_cast<double>(i) * 0.5 - 3.0);
            }

            const auto ug = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/false);
            const auto wg =
                CSRGraph<double>::build_from_weighted_edges(num_v, w_edges, /*directed=*/false);

            const auto u_res = find_biconnectivity(ug);
            const auto w_res = find_biconnectivity(wg);
            RC_ASSERT(w_res.articulation_points == u_res.articulation_points);
            RC_ASSERT(w_res.bridges == u_res.bridges);
            RC_ASSERT(canonicalize_components(w_res.biconnected_components) ==
                      canonicalize_components(u_res.biconnected_components));

            const auto u_bbt = build_bridge_block_tree(ug);
            const auto w_bbt = build_bridge_block_tree(wg);
            RC_ASSERT(w_bbt.vertex_to_component == u_bbt.vertex_to_component);
            RC_ASSERT(w_bbt.condensed_tree.num_vertices() == u_bbt.condensed_tree.num_vertices());
            RC_ASSERT(w_bbt.condensed_tree.num_edges() == u_bbt.condensed_tree.num_edges());

            const auto u_bct = build_block_cut_tree(ug);
            const auto w_bct = build_block_cut_tree(wg);
            RC_ASSERT(w_bct.articulation_points == u_bct.articulation_points);
            RC_ASSERT(w_bct.blocks == u_bct.blocks);
            RC_ASSERT(w_bct.tree.num_vertices() == u_bct.tree.num_vertices());
            RC_ASSERT(w_bct.tree.num_edges() == u_bct.tree.num_edges());
        });
}

TEST(BiconnectivityPBT, DirectedGraphRejected) {
    rc::check(
        "Directed CSRGraph inputs always throw std::invalid_argument",
        [](std::size_t raw_v, const std::vector<std::pair<uint32_t, uint32_t>>& raw_edges) {
            const std::size_t num_v = raw_v % 32U;
            const auto edges = normalize_edges(num_v, raw_edges);
            const auto dg = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/true);

            RC_ASSERT_THROWS_AS(static_cast<void>(find_biconnectivity(dg)), std::invalid_argument);
            RC_ASSERT_THROWS_AS(static_cast<void>(build_bridge_block_tree(dg)),
                                std::invalid_argument);
            RC_ASSERT_THROWS_AS(static_cast<void>(build_block_cut_tree(dg)), std::invalid_argument);
        });
}

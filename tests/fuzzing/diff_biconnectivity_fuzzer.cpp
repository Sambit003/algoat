#include "../graph/biconnectivity_test_utils.hpp"
#include "algoat/graph/biconnectivity.hpp"
#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

inline void fuzz_require(bool condition) noexcept {
    if (!condition) {
        std::abort();
    }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size == 0U) {
        return 0;
    }

    const std::size_t num_vertices = static_cast<std::size_t>(data[0]) % 49U; // [0, 48]
    const std::size_t remaining = size - 1U;
    const std::size_t num_edges = std::min<std::size_t>(remaining / 4U, 256U);

    std::vector<std::pair<VertexId, VertexId>> edges;
    if (num_vertices > 0U && num_edges > 0U) {
        edges.reserve(num_edges);
        for (std::size_t i = 0; i < num_edges; ++i) {
            uint16_t raw_u = 0;
            uint16_t raw_v = 0;
            std::memcpy(&raw_u, data + 1U + i * 4U, sizeof(uint16_t));
            std::memcpy(&raw_v, data + 1U + i * 4U + 2U, sizeof(uint16_t));
            edges.emplace_back(static_cast<VertexId>(raw_u % num_vertices),
                               static_cast<VertexId>(raw_v % num_vertices));
        }
    }

    const auto graph = CSRGraph<void>::build_from_edges(num_vertices, edges, /*directed=*/false);
    const BiconnectivityResult res = find_biconnectivity(graph);
    const BridgeBlockTree bbt = build_bridge_block_tree(graph);
    const BlockCutTree bct = build_block_cut_tree(graph);

    fuzz_require(verify_all_biconnectivity_invariants(graph, res, bbt, bct));

    const auto nx_ref = NetworkXReference::compute(graph);
    fuzz_require(res.articulation_points == nx_ref.articulation_points);
    fuzz_require(res.bridges == nx_ref.bridges);
    fuzz_require(canonicalize_components(res.biconnected_components) ==
                 nx_ref.biconnected_components);
    fuzz_require(bridge_tree_vertex_groups(bbt) == nx_ref.bridge_components);

    if (num_vertices <= 20U) {
        fuzz_require(res.articulation_points == BruteForceOracle::articulation_points(graph));
        fuzz_require(res.bridges == BruteForceOracle::bridges(graph));
        fuzz_require(bridge_tree_vertex_groups(bbt) == BruteForceOracle::bridge_components(graph));
    }

    return 0;
}

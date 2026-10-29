#include "../graph/scc_test_utils.hpp"
#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/scc.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

using namespace algoat::graph;
using algoat::graph::test::KosarajuReference;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    constexpr std::size_t min_header = sizeof(uint8_t);
    if (size < min_header + sizeof(uint16_t) * 2) {
        return 0;
    }

    const std::size_t num_vertices = (static_cast<std::size_t>(data[0]) % 32) + 1; // 1 to 32

    const std::size_t offset = min_header;
    const std::size_t remaining_bytes = size - offset;
    std::size_t num_edges = remaining_bytes / 4; // 2 bytes u, 2 bytes v
    if (num_edges > 128) {
        num_edges = 128; // Bound iterations per cycle
    }

    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(num_edges);

    for (std::size_t i = 0; i < num_edges; ++i) {
        uint16_t raw_u = 0;
        uint16_t raw_v = 0;
        std::memcpy(&raw_u, data + offset + i * 4, sizeof(uint16_t));
        std::memcpy(&raw_v, data + offset + i * 4 + 2, sizeof(uint16_t));

        const VertexId u = static_cast<VertexId>(raw_u % num_vertices);
        const VertexId v = static_cast<VertexId>(raw_v % num_vertices);
        edges.emplace_back(u, v);
    }

    auto g = CSRGraph<void>::build_from_edges(num_vertices, edges, /*directed=*/true);
    StronglyConnectedComponents engine(g);

    auto res = engine.compute();

    assert(res.num_components > 0 && res.num_components <= num_vertices);
    assert(res.component_ids.size() == num_vertices);
    assert(res.components.size() == res.num_components);

    std::size_t total_members = 0;
    for (uint32_t c = 0; c < res.num_components; ++c) {
        assert(!res.components[c].empty());
        total_members += res.components[c].size();
        for (VertexId u : res.components[c]) {
            assert(res.component_ids[u] == c);
        }
    }
    assert(total_members == num_vertices);

    // Verify condensation DAG invariants
    auto dag = engine.condensation_graph(res);
    assert(dag.num_vertices() == res.num_components);
    assert(dag.is_directed());

    for (VertexId u = 0; u < dag.num_vertices(); ++u) {
        for (VertexId v : dag.out_neighbors(u)) {
            assert(u < v); // Strictly upper-triangular / topological ordering
        }
    }

    // Verify differential equivalence against Kosaraju reference
    uint32_t ref_num_scc = 0;
    auto ref_comp = KosarajuReference::compute(num_vertices, edges, ref_num_scc);
    assert(res.num_components == ref_num_scc);

    for (VertexId u = 0; u < num_vertices; ++u) {
        for (VertexId v = u + 1; v < num_vertices; ++v) {
            bool same_in_engine = (res.component_ids[u] == res.component_ids[v]);
            bool same_in_ref = (ref_comp[u] == ref_comp[v]);
            assert(same_in_engine == same_in_ref);
        }
    }

    return 0;
}

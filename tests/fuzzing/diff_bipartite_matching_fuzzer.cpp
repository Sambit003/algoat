#include "algoat/graph/bipartite_matching.hpp"
#include "algoat/graph/max_flow.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

using namespace algoat::graph;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    constexpr std::size_t min_header = sizeof(uint8_t) * 2;
    if (size < min_header + sizeof(uint16_t) * 2) {
        return 0;
    }

    const std::size_t num_left = (static_cast<std::size_t>(data[0]) % 32) + 1;  // 1 to 32
    const std::size_t num_right = (static_cast<std::size_t>(data[1]) % 32) + 1; // 1 to 32

    std::size_t offset = min_header;
    const std::size_t remaining_bytes = size - offset;
    std::size_t num_edges = remaining_bytes / 4; // 2 bytes u, 2 bytes v
    if (num_edges > 128) {
        num_edges = 128; // Bound iterations per fuzz cycle
    }

    BipartiteGraph graph(num_left, num_right);

    // Setup Dinic unit flow reduction: source s = num_left + num_right, sink t = num_left +
    // num_right + 1
    const VertexId s = static_cast<VertexId>(num_left + num_right);
    const VertexId t = static_cast<VertexId>(num_left + num_right + 1);
    MaxFlowEngine<int64_t> flow_engine(num_left + num_right + 2);

    for (VertexId u = 0; u < num_left; ++u) {
        flow_engine.add_edge(s, u, 1);
    }
    for (VertexId v = 0; v < num_right; ++v) {
        flow_engine.add_edge(static_cast<VertexId>(num_left + v), t, 1);
    }

    std::unordered_set<uint64_t> added_edges;
    for (std::size_t i = 0; i < num_edges; ++i) {
        uint16_t raw_u = 0;
        uint16_t raw_v = 0;
        std::memcpy(&raw_u, data + offset + i * 4, sizeof(uint16_t));
        std::memcpy(&raw_v, data + offset + i * 4 + 2, sizeof(uint16_t));

        const VertexId u = static_cast<VertexId>(raw_u % num_left);
        const VertexId v = static_cast<VertexId>(raw_v % num_right);
        const uint64_t key = (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);

        if (added_edges.insert(key).second) {
            graph.add_edge(u, v);
            flow_engine.add_edge(u, static_cast<VertexId>(num_left + v), 1);
        }
    }

    const auto matching = graph.maximum_matching();
    const auto dinic_res = flow_engine.dinic(s, t);

    // Differential cross-check against Dinic max-flow
    assert(matching.matching_size == static_cast<std::size_t>(dinic_res.max_flow));

    // König's theorem duality verification
    const auto [cover_l, cover_r] = graph.minimum_vertex_cover(matching);
    assert(cover_l.size() + cover_r.size() == matching.matching_size);

    std::vector<bool> in_cover_l(num_left, false);
    for (VertexId u : cover_l)
        in_cover_l[u] = true;
    std::vector<bool> in_cover_r(num_right, false);
    for (VertexId v : cover_r)
        in_cover_r[v] = true;

    for (VertexId u = 0; u < num_left; ++u) {
        for (VertexId v : graph.adj(u)) {
            assert(in_cover_l[u] || in_cover_r[v]);
        }
    }

    // Maximum independent set verification
    const auto [indep_l, indep_r] = graph.maximum_independent_set(matching);
    assert(indep_l.size() + indep_r.size() == num_left + num_right - matching.matching_size);

    // Berge's lemma invariant verification
    assert(BipartiteMatchingTestAccess::verify_berge_invariant(graph, matching));

    return 0;
}

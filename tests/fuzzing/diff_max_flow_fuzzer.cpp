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
    constexpr std::size_t min_header = sizeof(uint16_t) + 2 * sizeof(uint8_t);
    if (size < min_header + sizeof(uint16_t) * 3) {
        return 0;
    }

    uint16_t raw_v = 0;
    std::memcpy(&raw_v, data, sizeof(uint16_t));
    const std::size_t num_vertices = (static_cast<std::size_t>(raw_v) % 32) + 2; // 2 to 33 vertices

    const VertexId source = static_cast<VertexId>(data[2]) % num_vertices;
    const VertexId sink = static_cast<VertexId>(data[3]) % num_vertices;

    if (source == sink) {
        return 0;
    }

    std::size_t offset = min_header;
    const std::size_t remaining_bytes = size - offset;
    std::size_t num_edges = remaining_bytes / 6; // 2 bytes u, 2 bytes v, 2 bytes capacity
    if (num_edges > 128) {
        num_edges = 128; // Bound work per cycle
    }

    MaxFlowEngine<int64_t> engine(num_vertices);

    for (std::size_t i = 0; i < num_edges; ++i) {
        uint16_t raw_u = 0;
        uint16_t raw_v_edge = 0;
        uint16_t raw_cap = 0;
        std::memcpy(&raw_u, data + offset + i * 6, sizeof(uint16_t));
        std::memcpy(&raw_v_edge, data + offset + i * 6 + 2, sizeof(uint16_t));
        std::memcpy(&raw_cap, data + offset + i * 6 + 4, sizeof(uint16_t));

        const VertexId u = static_cast<VertexId>(raw_u % num_vertices);
        const VertexId v = static_cast<VertexId>(raw_v_edge % num_vertices);
        const int64_t cap = static_cast<int64_t>((raw_cap % 1000) + 1);

        engine.add_edge(u, v, cap);
    }

    auto res_dinic = engine.dinic(source, sink);
    auto res_pr = engine.push_relabel(source, sink);

    assert(res_dinic.max_flow == res_pr.max_flow);
    assert(res_dinic.max_flow >= 0);

    // Verify Dinic flow invariants
    std::vector<int64_t> net_flow(num_vertices, 0);
    for (const auto& e : res_dinic.residual_edges) {
        assert(e.flow >= 0);
        assert(e.flow <= e.capacity);
        net_flow[e.from] -= e.flow;
        net_flow[e.to] += e.flow;
    }

    assert(-net_flow[source] == res_dinic.max_flow);
    assert(net_flow[sink] == res_dinic.max_flow);
    for (VertexId v = 0; v < num_vertices; ++v) {
        if (v != source && v != sink) {
            assert(net_flow[v] == 0);
        }
    }

    // Verify Dinic Min-Cut
    std::unordered_set<VertexId> s_set(res_dinic.min_cut_source_set.begin(),
                                       res_dinic.min_cut_source_set.end());
    assert(s_set.contains(source));
    assert(!s_set.contains(sink));

    int64_t cut_cap = 0;
    for (const auto& e : res_dinic.residual_edges) {
        if (s_set.contains(e.from) && !s_set.contains(e.to)) {
            cut_cap += e.capacity;
        }
    }
    assert(cut_cap == res_dinic.max_flow);

    // Verify Push-Relabel flow invariants
    std::fill(net_flow.begin(), net_flow.end(), 0);
    for (const auto& e : res_pr.residual_edges) {
        assert(e.flow >= 0);
        assert(e.flow <= e.capacity);
        net_flow[e.from] -= e.flow;
        net_flow[e.to] += e.flow;
    }

    assert(-net_flow[source] == res_pr.max_flow);
    assert(net_flow[sink] == res_pr.max_flow);
    for (VertexId v = 0; v < num_vertices; ++v) {
        if (v != source && v != sink) {
            assert(net_flow[v] == 0);
        }
    }

    return 0;
}

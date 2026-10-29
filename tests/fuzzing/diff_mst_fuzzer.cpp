#include "algoat/graph/mst.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

using namespace algoat::graph;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    constexpr std::size_t min_header = sizeof(uint8_t) + sizeof(uint8_t);
    if (size < min_header + sizeof(uint16_t) * 3) {
        return 0;
    }

    const std::size_t num_vertices = (static_cast<std::size_t>(data[0]) % 48) + 1; // 1 to 48
    const std::size_t threads = (static_cast<std::size_t>(data[1]) % 4) + 1;       // 1 to 4

    std::size_t offset = min_header;
    const std::size_t remaining_bytes = size - offset;
    std::size_t num_edges = remaining_bytes / 6; // 2 bytes u, 2 bytes v, 2 bytes w
    if (num_edges > 128) {
        num_edges = 128; // Bound iterations per cycle
    }

    std::vector<std::tuple<VertexId, VertexId, double>> edges;
    edges.reserve(num_edges);

    for (std::size_t i = 0; i < num_edges; ++i) {
        uint16_t raw_u = 0;
        uint16_t raw_v = 0;
        uint16_t raw_w = 0;
        std::memcpy(&raw_u, data + offset + i * 6, sizeof(uint16_t));
        std::memcpy(&raw_v, data + offset + i * 6 + 2, sizeof(uint16_t));
        std::memcpy(&raw_w, data + offset + i * 6 + 4, sizeof(uint16_t));

        const VertexId u = static_cast<VertexId>(raw_u % num_vertices);
        const VertexId v = static_cast<VertexId>(raw_v % num_vertices);
        // Map weights into range [-200.0, 200.0]
        const double w = static_cast<double>(static_cast<int32_t>(raw_w % 401) - 200);
        edges.emplace_back(u, v, w);
    }

    auto g = CSRGraph<double>::build_from_weighted_edges(num_vertices, edges, /*directed=*/false);
    MinimumSpanningTreeEngine<double> engine(g);

    auto serial_res = engine.compute_serial();
    auto parallel_res = engine.compute_parallel(threads);

    assert(std::abs(serial_res.total_weight - parallel_res.total_weight) < 1e-5);
    assert(serial_res.components_count == parallel_res.components_count);
    assert(serial_res.edges.size() == parallel_res.edges.size());
    assert(serial_res.edges.size() + serial_res.components_count == num_vertices);

    // Verify acyclic property on parallel output
    DisjointSetUnion dsu(num_vertices);
    for (const auto& edge : parallel_res.edges) {
        bool united = dsu.unite(edge.u, edge.v);
        assert(united);
    }
    assert(dsu.num_components() == parallel_res.components_count);

    return 0;
}

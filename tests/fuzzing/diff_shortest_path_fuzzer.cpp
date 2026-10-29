#include "algoat/graph/shortest_path.hpp"

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
    constexpr std::size_t min_header =
        sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t) + 2 * sizeof(uint8_t);
    if (size < min_header + sizeof(uint16_t) * 3) {
        return 0;
    }

    const std::size_t num_vertices = (static_cast<std::size_t>(data[0]) % 32) + 2; // 2 to 33
    const std::size_t num_landmarks = (static_cast<std::size_t>(data[1]) % 8) + 1; // 1 to 8
    const bool directed = (data[2] & 1) != 0;
    const VertexId source = static_cast<VertexId>(data[3]) % num_vertices;
    const VertexId target = static_cast<VertexId>(data[4]) % num_vertices;

    std::size_t offset = min_header;
    const std::size_t remaining_bytes = size - offset;
    std::size_t num_edges = remaining_bytes / 6; // 2 bytes u, 2 bytes v, 2 bytes w
    if (num_edges > 128) {
        num_edges = 128; // Bound per cycle
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
        const double w = static_cast<double>((raw_w % 500) + 1); // positive weight
        edges.emplace_back(u, v, w);
    }

    auto g = CSRGraph<double>::build_from_weighted_edges(num_vertices, edges, directed);
    ShortestPathEngine<double> engine(g);
    engine.preprocess_landmarks(num_landmarks);

    std::vector<VertexId> p1;
    std::vector<VertexId> p2;
    std::vector<VertexId> p3;
    std::vector<VertexId> p4;

    auto d1 = engine.dijkstra_p2p(source, target, &p1);
    auto d2 = engine.alt_search(source, target, &p2);
    auto d3 = engine.dijkstra_bidirectional(source, target, &p3);
    auto d4 = engine.alt_search_bidirectional(source, target, &p4);

    assert(d1.has_value() == d2.has_value());
    assert(d1.has_value() == d3.has_value());
    assert(d1.has_value() == d4.has_value());

    if (d1.has_value()) {
        assert(std::abs(*d1 - *d2) < 1e-5);
        assert(std::abs(*d1 - *d3) < 1e-5);
        assert(std::abs(*d1 - *d4) < 1e-5);

        if (!p1.empty()) {
            assert(p1.front() == source && p1.back() == target);
        }
        if (!p2.empty()) {
            assert(p2.front() == source && p2.back() == target);
        }
        if (!p3.empty()) {
            assert(p3.front() == source && p3.back() == target);
        }
        if (!p4.empty()) {
            assert(p4.front() == source && p4.back() == target);
        }
    }

    return 0;
}

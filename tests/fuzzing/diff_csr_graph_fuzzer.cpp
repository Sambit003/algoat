#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace algoat::graph;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    // Need at least 4 bytes for num_vertices, 1 byte for directed flag, and at least one edge (8
    // bytes)
    constexpr std::size_t min_header = sizeof(uint16_t) + sizeof(uint8_t);
    if (size < min_header + sizeof(uint32_t) * 2) {
        return 0;
    }

    uint16_t raw_v = 0;
    std::memcpy(&raw_v, data, sizeof(uint16_t));
    std::size_t num_vertices = (raw_v % 128) + 1; // 1 to 128 vertices

    bool directed = (data[sizeof(uint16_t)] & 1) != 0;

    std::size_t offset = min_header;
    std::size_t remaining_bytes = size - offset;
    std::size_t num_pairs = remaining_bytes / (2 * sizeof(uint32_t));
    if (num_pairs > 256) {
        num_pairs = 256; // Bound work per fuzz cycle
    }

    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(num_pairs);

    for (std::size_t i = 0; i < num_pairs; ++i) {
        uint32_t u = 0;
        uint32_t v = 0;
        std::memcpy(&u, data + offset + i * 8, sizeof(uint32_t));
        std::memcpy(&v, data + offset + i * 8 + 4, sizeof(uint32_t));
        edges.emplace_back(u % num_vertices, v % num_vertices);
    }

    // Build CSRGraph
    CSRGraph g = CSRGraph<>::build_from_edges(num_vertices, edges, directed);

    // Build Reference Oracle using naive vector-of-vectors
    std::vector<std::vector<VertexId>> oracle_out(num_vertices);
    std::vector<std::vector<VertexId>> oracle_in(num_vertices);

    for (const auto& [u, v] : edges) {
        if (directed) {
            oracle_out[u].push_back(v);
            oracle_in[v].push_back(u);
        } else {
            oracle_out[u].push_back(v);
            if (u != v) {
                oracle_out[v].push_back(u);
            }
        }
    }

    for (std::size_t u = 0; u < num_vertices; ++u) {
        std::sort(oracle_out[u].begin(), oracle_out[u].end());
        auto u_end = std::unique(oracle_out[u].begin(), oracle_out[u].end());
        oracle_out[u].erase(u_end, oracle_out[u].end());

        if (directed) {
            std::sort(oracle_in[u].begin(), oracle_in[u].end());
            auto v_end = std::unique(oracle_in[u].begin(), oracle_in[u].end());
            oracle_in[u].erase(v_end, oracle_in[u].end());
        }
    }

    // Differential Verification against Reference Oracle
    if (g.num_vertices() != num_vertices) {
        __builtin_trap();
    }

    for (VertexId u = 0; u < num_vertices; ++u) {
        if (g.out_degree(u) != oracle_out[u].size()) {
            __builtin_trap();
        }
        auto out_span = g.out_neighbors(u);
        for (std::size_t i = 0; i < oracle_out[u].size(); ++i) {
            if (out_span[i] != oracle_out[u][i]) {
                __builtin_trap();
            }
        }

        if (directed) {
            if (g.in_degree(u) != oracle_in[u].size()) {
                __builtin_trap();
            }
            auto in_span = g.in_neighbors(u);
            for (std::size_t i = 0; i < oracle_in[u].size(); ++i) {
                if (in_span[i] != oracle_in[u][i]) {
                    __builtin_trap();
                }
            }
        } else {
            // Undirected: in_neighbors equals out_neighbors
            if (g.in_degree(u) != oracle_out[u].size()) {
                __builtin_trap();
            }
            auto in_span = g.in_neighbors(u);
            for (std::size_t i = 0; i < oracle_out[u].size(); ++i) {
                if (in_span[i] != oracle_out[u][i]) {
                    __builtin_trap();
                }
            }
        }
    }

    // Transposition Differential Check
    CSRGraph gt = g.transpose();
    if (directed) {
        for (VertexId u = 0; u < num_vertices; ++u) {
            if (gt.out_degree(u) != oracle_in[u].size()) {
                __builtin_trap();
            }
            auto gt_out = gt.out_neighbors(u);
            for (std::size_t i = 0; i < oracle_in[u].size(); ++i) {
                if (gt_out[i] != oracle_in[u][i]) {
                    __builtin_trap();
                }
            }
        }
    }

    return 0;
}

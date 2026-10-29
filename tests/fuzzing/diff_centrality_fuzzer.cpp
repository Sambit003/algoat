#include "algoat/graph/centrality.hpp"
#include "algoat/graph/csr_graph.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

using namespace algoat::graph;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    constexpr std::size_t min_header = sizeof(uint16_t) + sizeof(uint8_t);
    if (size < min_header + sizeof(uint32_t) * 2) {
        return 0;
    }

    uint16_t raw_v = 0;
    std::memcpy(&raw_v, data, sizeof(uint16_t));
    std::size_t num_vertices = (raw_v % 64) + 1; // 1 to 64 vertices

    bool directed = (data[sizeof(uint16_t)] & 1) != 0;

    std::size_t offset = min_header;
    std::size_t remaining_bytes = size - offset;
    std::size_t num_pairs = remaining_bytes / (2 * sizeof(uint32_t));
    if (num_pairs > 128) {
        num_pairs = 128; // Bound work per fuzz cycle
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
    auto g = CSRGraph<void>::build_from_edges(num_vertices, edges, directed);

    // Exercise PageRank
    PageRankConfig pr_cfg;
    pr_cfg.damping_factor = 0.85;
    pr_cfg.tolerance = 1e-5;
    pr_cfg.max_iterations = 50;
    auto pr_res = pagerank(g, pr_cfg);

    assert(pr_res.scores.size() == num_vertices);
    double pr_sum = std::accumulate(pr_res.scores.begin(), pr_res.scores.end(), 0.0);
    assert(std::abs(pr_sum - 1.0) < 1e-8);

    // Exercise Betweenness Centrality
    auto bc_res = betweenness_centrality(g);
    assert(bc_res.betweenness.size() == num_vertices);
    assert(bc_res.degree_centrality.size() == num_vertices);

    for (std::size_t i = 0; i < num_vertices; ++i) {
        assert(bc_res.betweenness[i] >= 0.0);
        assert(!std::isnan(bc_res.betweenness[i]));
        assert(bc_res.degree_centrality[i] >= 0.0);
    }

    return 0;
}

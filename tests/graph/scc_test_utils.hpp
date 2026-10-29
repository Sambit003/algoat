/**
 * @file scc_test_utils.hpp
 * @brief Shared independent reference implementations and graph generators for SCC testing.
 */

#pragma once

#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace algoat::graph::test {

/**
 * @struct KosarajuReference
 * @brief Independent textbook reference implementation of Kosaraju's two-pass SCC algorithm.
 *
 * Used for differential fuzzing and property-based verification against
 * StronglyConnectedComponents.
 */
struct KosarajuReference {
    [[nodiscard]] static std::vector<uint32_t>
    compute(std::size_t n, std::span<const std::pair<VertexId, VertexId>> edges,
            uint32_t& num_scc) {
        if (n == 0) {
            num_scc = 0;
            return {};
        }

        std::vector<std::vector<VertexId>> adj(n);
        std::vector<std::vector<VertexId>> rev_adj(n);
        for (const auto& [u, v] : edges) {
            if (u < n && v < n) {
                adj[u].push_back(v);
                rev_adj[v].push_back(u);
            }
        }

        std::vector<bool> visited(n, false);
        std::vector<VertexId> finish_order;
        finish_order.reserve(n);

        // Pass 1: Forward DFS to record finish order
        auto dfs_forward = [&](auto& self, VertexId u) -> void {
            visited[u] = true;
            for (VertexId v : adj[u]) {
                if (!visited[v]) {
                    self(self, v);
                }
            }
            finish_order.push_back(u);
        };

        for (VertexId i = 0; i < n; ++i) {
            if (!visited[i]) {
                dfs_forward(dfs_forward, i);
            }
        }

        // Pass 2: Reverse DFS on transposed graph in decreasing finish order
        std::fill(visited.begin(), visited.end(), false);
        std::vector<uint32_t> component_map(n, 0);
        num_scc = 0;

        auto dfs_reverse = [&](auto& self, VertexId u, uint32_t comp_id) -> void {
            visited[u] = true;
            component_map[u] = comp_id;
            for (VertexId v : rev_adj[u]) {
                if (!visited[v]) {
                    self(self, v, comp_id);
                }
            }
        };

        for (std::size_t i = n; i > 0; --i) {
            VertexId u = finish_order[i - 1];
            if (!visited[u]) {
                dfs_reverse(dfs_reverse, u, num_scc++);
            }
        }

        return component_map;
    }
};

} // namespace algoat::graph::test

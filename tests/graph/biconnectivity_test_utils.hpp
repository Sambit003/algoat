/**
 * @file biconnectivity_test_utils.hpp
 * @brief Independent reference oracles (brute-force connectivity deletion, NetworkX-equivalent
 * edge-stack _biconnected_dfs, and Schmidt 2013 chain decomposition), graph generators, and
 * structural invariant verifiers for biconnectivity testing.
 */

#pragma once

#include "algoat/graph/biconnectivity.hpp"
#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace algoat::graph::test {

[[nodiscard]] inline std::vector<std::vector<std::size_t>>
canonicalize_components(std::vector<std::vector<std::size_t>> comps) {
    for (auto& comp : comps) {
        std::sort(comp.begin(), comp.end());
    }
    std::sort(comps.begin(), comps.end());
    return comps;
}

[[nodiscard]] inline std::vector<std::vector<VertexId>>
extract_simple_adj(const CSRGraph<void>& graph) {
    const std::size_t n = graph.num_vertices();
    std::vector<std::vector<VertexId>> adj(n);
    for (VertexId u = 0; u < static_cast<VertexId>(n); ++u) {
        for (const VertexId v : graph.out_neighbors(u)) {
            if (u != v) {
                adj[u].push_back(v);
            }
        }
    }
    return adj;
}

/**
 * @struct BruteForceOracle
 * @brief Independent definition-based oracle computing articulation points, bridges, and
 * 2-edge-connected components directly via vertex/edge deletion and BFS component counting.
 */
struct BruteForceOracle {
    static std::size_t
    count_components(std::size_t n, const std::vector<std::vector<VertexId>>& adj,
                     std::size_t skip_vertex = std::numeric_limits<std::size_t>::max(),
                     std::size_t skip_u = std::numeric_limits<std::size_t>::max(),
                     std::size_t skip_v = std::numeric_limits<std::size_t>::max()) {
        std::vector<uint8_t> visited(n, 0);
        std::queue<VertexId> q;
        std::size_t components = 0;

        for (std::size_t start = 0; start < n; ++start) {
            if (start == skip_vertex || visited[start] != 0) {
                continue;
            }
            ++components;
            visited[start] = 1;
            q.push(static_cast<VertexId>(start));

            while (!q.empty()) {
                const VertexId u = q.front();
                q.pop();
                for (const VertexId v : adj[u]) {
                    if (v == skip_vertex || visited[v] != 0) {
                        continue;
                    }
                    if ((u == skip_u && v == skip_v) || (u == skip_v && v == skip_u)) {
                        continue;
                    }
                    visited[v] = 1;
                    q.push(v);
                }
            }
        }
        return components;
    }

    [[nodiscard]] static std::vector<std::size_t> articulation_points(const CSRGraph<void>& graph) {
        const std::size_t n = graph.num_vertices();
        const auto adj = extract_simple_adj(graph);
        const std::size_t base_cc = count_components(n, adj);

        std::vector<std::size_t> cuts;
        for (std::size_t v = 0; v < n; ++v) {
            if (count_components(n, adj, v) > base_cc) {
                cuts.push_back(v);
            }
        }
        return cuts;
    }

    [[nodiscard]] static std::vector<std::pair<std::size_t, std::size_t>>
    bridges(const CSRGraph<void>& graph) {
        const std::size_t n = graph.num_vertices();
        const auto adj = extract_simple_adj(graph);
        const std::size_t base_cc = count_components(n, adj);

        std::vector<std::pair<std::size_t, std::size_t>> result;
        for (std::size_t u = 0; u < n; ++u) {
            for (const VertexId v : adj[u]) {
                if (u < v && count_components(n, adj, std::numeric_limits<std::size_t>::max(), u,
                                              v) > base_cc) {
                    result.emplace_back(u, v);
                }
            }
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    [[nodiscard]] static std::vector<std::vector<std::size_t>>
    bridge_components(const CSRGraph<void>& graph) {
        const std::size_t n = graph.num_vertices();
        const auto adj = extract_simple_adj(graph);
        auto b_list = bridges(graph);

        std::vector<uint8_t> visited(n, 0);
        std::queue<VertexId> q;
        std::vector<std::vector<std::size_t>> comps;

        for (std::size_t start = 0; start < n; ++start) {
            if (visited[start] != 0) {
                continue;
            }
            std::vector<std::size_t> comp;
            visited[start] = 1;
            q.push(static_cast<VertexId>(start));

            while (!q.empty()) {
                const VertexId u = q.front();
                q.pop();
                comp.push_back(u);
                for (const VertexId v : adj[u]) {
                    if (visited[v] != 0) {
                        continue;
                    }
                    const std::pair<std::size_t, std::size_t> edge = {std::min<std::size_t>(u, v),
                                                                      std::max<std::size_t>(u, v)};
                    if (std::binary_search(b_list.begin(), b_list.end(), edge)) {
                        continue;
                    }
                    visited[v] = 1;
                    q.push(v);
                }
            }
            std::sort(comp.begin(), comp.end());
            comps.push_back(std::move(comp));
        }
        std::sort(comps.begin(), comps.end());
        return comps;
    }
};

/**
 * @struct NetworkXReference
 * @brief Faithful non-recursive C++ reference implementing NetworkX semantics:
 *  - `_biconnected_dfs` (edge-stack + frame edge_index) for articulation_points &
 *    biconnected_components
 *  - Schmidt (2013) `chain_decomposition` (NetworkX `bridges.py`) for bridges & bridge_components
 */
struct NetworkXReference {
    struct BiconOutput {
        std::vector<std::size_t> articulation_points;
        std::vector<std::pair<std::size_t, std::size_t>> bridges;
        std::vector<std::vector<std::size_t>> biconnected_components;
        std::vector<std::vector<std::size_t>> bridge_components;
    };

    [[nodiscard]] static BiconOutput compute(const CSRGraph<void>& graph) {
        BiconOutput out;
        const std::size_t n = graph.num_vertices();
        if (n == 0) {
            return out;
        }

        // 1. NetworkX `_biconnected_dfs`: edge-stack formulation
        {
            constexpr uint32_t kUnvisited = std::numeric_limits<uint32_t>::max();
            std::vector<uint32_t> discovery(n, kUnvisited);
            std::vector<uint32_t> low(n, 0);
            std::vector<uint8_t> is_art(n, 0);
            std::vector<uint32_t> seen_mark(n, 0);
            uint32_t mark_epoch = 0;

            struct NxFrame {
                VertexId grandparent;
                VertexId parent;
                std::size_t next_idx;
                std::size_t edge_stack_pos;
            };
            std::vector<NxFrame> stack;
            std::vector<std::pair<VertexId, VertexId>> edge_stack;

            auto pop_biconnected_component = [&](std::size_t pop_pos) {
                ++mark_epoch;
                std::vector<std::size_t> comp;
                for (std::size_t i = pop_pos; i < edge_stack.size(); ++i) {
                    const auto [eu, ev] = edge_stack[i];
                    if (seen_mark[eu] != mark_epoch) {
                        seen_mark[eu] = mark_epoch;
                        comp.push_back(eu);
                    }
                    if (seen_mark[ev] != mark_epoch) {
                        seen_mark[ev] = mark_epoch;
                        comp.push_back(ev);
                    }
                }
                edge_stack.resize(pop_pos);
                std::sort(comp.begin(), comp.end());
                out.biconnected_components.push_back(std::move(comp));
            };

            for (VertexId start = 0; start < static_cast<VertexId>(n); ++start) {
                if (discovery[start] != kUnvisited) {
                    continue;
                }
                uint32_t disc_counter = 0;
                discovery[start] = low[start] = disc_counter++;
                uint32_t root_children = 0;
                stack.push_back(NxFrame{start, start, 0, 0});

                while (!stack.empty()) {
                    NxFrame& top = stack.back();
                    const VertexId grandparent = top.grandparent;
                    const VertexId parent = top.parent;
                    const auto neigh = graph.out_neighbors(parent);

                    if (top.next_idx < neigh.size()) {
                        const VertexId child = neigh[top.next_idx++];
                        if (child == parent || child == grandparent) {
                            continue;
                        }
                        if (discovery[child] != kUnvisited) {
                            if (discovery[child] <= discovery[parent]) {
                                low[parent] = std::min(low[parent], discovery[child]);
                                edge_stack.emplace_back(parent, child);
                            }
                        } else {
                            low[child] = discovery[child] = disc_counter++;
                            const std::size_t pos = edge_stack.size();
                            edge_stack.emplace_back(parent, child);
                            stack.push_back(NxFrame{parent, child, 0, pos});
                        }
                    } else {
                        const std::size_t pop_pos = top.edge_stack_pos;
                        stack.pop_back();
                        if (stack.size() > 1U) {
                            if (low[parent] >= discovery[grandparent]) {
                                is_art[grandparent] = 1;
                                pop_biconnected_component(pop_pos);
                            }
                            low[grandparent] = std::min(low[parent], low[grandparent]);
                        } else if (!stack.empty()) {
                            ++root_children;
                            pop_biconnected_component(pop_pos);
                        }
                    }
                }
                if (root_children > 1U) {
                    is_art[start] = 1;
                }
            }

            for (std::size_t u = 0; u < n; ++u) {
                if (is_art[u] != 0) {
                    out.articulation_points.push_back(u);
                }
            }
            std::sort(out.biconnected_components.begin(), out.biconnected_components.end());
        }

        // 2. NetworkX `bridges.py` via Schmidt (2013) `chain_decomposition` (zero low[] usage)
        {
            constexpr VertexId kNoParent = std::numeric_limits<VertexId>::max();
            std::vector<uint32_t> tin(n, 0U);
            std::vector<VertexId> parent(n, kNoParent);
            std::vector<VertexId> preorder;
            preorder.reserve(n);
            std::vector<std::vector<VertexId>> back_edges_from(n);

            struct DfsState {
                VertexId u;
                std::size_t idx;
            };
            std::vector<DfsState> dfs_stack;
            uint32_t timer = 0;

            for (VertexId root = 0; root < static_cast<VertexId>(n); ++root) {
                if (tin[root] != 0U) {
                    continue;
                }
                tin[root] = ++timer;
                preorder.push_back(root);
                dfs_stack.push_back(DfsState{root, 0});

                while (!dfs_stack.empty()) {
                    DfsState& st = dfs_stack.back();
                    const VertexId u = st.u;
                    const auto neigh = graph.out_neighbors(u);
                    if (st.idx < neigh.size()) {
                        const VertexId v = neigh[st.idx++];
                        if (v == u || v == parent[u]) {
                            continue;
                        }
                        if (tin[v] == 0U) {
                            tin[v] = ++timer;
                            parent[v] = u;
                            preorder.push_back(v);
                            dfs_stack.push_back(DfsState{v, 0});
                        } else if (tin[v] < tin[u]) {
                            // Back-edge oriented from ancestor v down to descendant u
                            back_edges_from[v].push_back(u);
                        }
                    } else {
                        dfs_stack.pop_back();
                    }
                }
            }

            std::vector<uint8_t> visited_chain(n, 0);
            std::vector<uint8_t> tree_edge_in_chain(n, 0);
            for (const VertexId a : preorder) {
                for (const VertexId d : back_edges_from[a]) {
                    visited_chain[a] = 1;
                    VertexId cur = d;
                    while (visited_chain[cur] == 0) {
                        visited_chain[cur] = 1;
                        tree_edge_in_chain[cur] = 1;
                        cur = parent[cur];
                    }
                }
            }

            for (VertexId u = 0; u < static_cast<VertexId>(n); ++u) {
                if (parent[u] != kNoParent && tree_edge_in_chain[u] == 0) {
                    out.bridges.emplace_back(std::min<std::size_t>(u, parent[u]),
                                             std::max<std::size_t>(u, parent[u]));
                }
            }
            std::sort(out.bridges.begin(), out.bridges.end());

            // 3. `bridge_components`: BFS over G \ bridges (marked via tree_edge_in_chain)
            std::vector<uint8_t> visited_bfs(n, 0);
            std::queue<VertexId> q;
            for (VertexId start = 0; start < static_cast<VertexId>(n); ++start) {
                if (visited_bfs[start] != 0) {
                    continue;
                }
                std::vector<std::size_t> comp;
                visited_bfs[start] = 1;
                q.push(start);
                while (!q.empty()) {
                    const VertexId u = q.front();
                    q.pop();
                    comp.push_back(u);
                    for (const VertexId v : graph.out_neighbors(u)) {
                        if (u == v || visited_bfs[v] != 0) {
                            continue;
                        }
                        const bool is_bridge_uv = (parent[v] == u && tree_edge_in_chain[v] == 0) ||
                                                  (parent[u] == v && tree_edge_in_chain[u] == 0);
                        if (is_bridge_uv) {
                            continue;
                        }
                        visited_bfs[v] = 1;
                        q.push(v);
                    }
                }
                std::sort(comp.begin(), comp.end());
                out.bridge_components.push_back(std::move(comp));
            }
            std::sort(out.bridge_components.begin(), out.bridge_components.end());
        }

        return out;
    }
};

/**
 * @brief Extracts canonical 2-ECC vertex groups from a BridgeBlockTree result.
 */
[[nodiscard]] inline std::vector<std::vector<std::size_t>>
bridge_tree_vertex_groups(const BridgeBlockTree& bbt) {
    const std::size_t num_c = bbt.condensed_tree.num_vertices();
    std::vector<std::vector<std::size_t>> groups(num_c);
    for (std::size_t v = 0; v < bbt.vertex_to_component.size(); ++v) {
        const std::size_t cid = bbt.vertex_to_component[v];
        if (cid >= num_c) {
            return {};
        }
        groups[cid].push_back(v);
    }
    return canonicalize_components(std::move(groups));
}

/**
 * @brief Verifies structural acyclicity of an undirected CSRGraph forest and returns its edge
 * count.
 */
[[nodiscard]] inline bool is_undirected_forest(const CSRGraph<void>& forest,
                                               std::size_t* out_undirected_edges = nullptr) {
    const std::size_t n = forest.num_vertices();
    if (forest.is_directed()) {
        return false;
    }
    std::vector<uint8_t> visited(n, 0);
    std::queue<std::pair<VertexId, VertexId>> q;
    std::size_t edge_sum = 0;
    std::size_t cc_count = 0;

    for (VertexId start = 0; start < static_cast<VertexId>(n); ++start) {
        edge_sum += forest.out_degree(start);
        if (visited[start] != 0) {
            continue;
        }
        ++cc_count;
        visited[start] = 1;
        q.emplace(start, std::numeric_limits<VertexId>::max());
        while (!q.empty()) {
            const auto [u, parent] = q.front();
            q.pop();
            for (const VertexId v : forest.out_neighbors(u)) {
                if (v == u) {
                    return false;
                }
                if (v == parent) {
                    continue;
                }
                if (visited[v] != 0) {
                    return false; // Cycle detected
                }
                visited[v] = 1;
                q.emplace(v, u);
            }
        }
    }

    if ((edge_sum % 2U) != 0U) {
        return false;
    }
    const std::size_t undirected_edges = edge_sum / 2U;
    if (out_undirected_edges != nullptr) {
        *out_undirected_edges = undirected_edges;
    }
    return n == 0 ? undirected_edges == 0 : (undirected_edges + cc_count == n);
}

/**
 * @brief Verifies all structural invariants of BiconnectivityResult, BridgeBlockTree, and
 * BlockCutTree for any undirected CSRGraph (safe under both Debug and Release/NDEBUG).
 */
[[nodiscard]] inline bool verify_all_biconnectivity_invariants(const CSRGraph<void>& graph,
                                                               const BiconnectivityResult& res,
                                                               const BridgeBlockTree& bbt,
                                                               const BlockCutTree& bct) {
    const std::size_t n = graph.num_vertices();

    // 1. Articulation points: sorted, unique, < n
    if (!std::is_sorted(res.articulation_points.begin(), res.articulation_points.end()) ||
        std::adjacent_find(res.articulation_points.begin(), res.articulation_points.end()) !=
            res.articulation_points.end()) {
        return false;
    }
    for (const std::size_t ap : res.articulation_points) {
        if (ap >= n) {
            return false;
        }
    }

    // 2. Bridges: normalized u < v, sorted, unique, exist in graph
    if (!std::is_sorted(res.bridges.begin(), res.bridges.end()) ||
        std::adjacent_find(res.bridges.begin(), res.bridges.end()) != res.bridges.end()) {
        return false;
    }
    for (const auto& [u, v] : res.bridges) {
        if (u >= v || v >= n ||
            !graph.find_out_edge_index(static_cast<VertexId>(u), static_cast<VertexId>(v))
                 .has_value()) {
            return false;
        }
    }

    // 3. Biconnected components: size >= 2, sorted unique vertices, intersection <= 1 (cut vertex)
    std::vector<std::size_t> block_membership_count(n, 0);
    std::size_t total_block_verts = 0;
    for (const auto& comp : res.biconnected_components) {
        if (comp.size() < 2U || !std::is_sorted(comp.begin(), comp.end()) ||
            std::adjacent_find(comp.begin(), comp.end()) != comp.end()) {
            return false;
        }
        for (const std::size_t v : comp) {
            if (v >= n) {
                return false;
            }
            ++block_membership_count[v];
        }
        total_block_verts += comp.size();
    }
    if (total_block_verts > 2U * n) {
        return false;
    }
    for (std::size_t v = 0; v < n; ++v) {
        const bool is_ap =
            std::binary_search(res.articulation_points.begin(), res.articulation_points.end(), v);
        if (is_ap && block_membership_count[v] < 2U) {
            return false;
        }
        if (!is_ap && block_membership_count[v] > 1U) {
            return false;
        }
    }

    if (n <= 128U) {
        for (std::size_t i = 0; i < res.biconnected_components.size(); ++i) {
            for (std::size_t j = i + 1; j < res.biconnected_components.size(); ++j) {
                std::vector<std::size_t> inter;
                std::set_intersection(
                    res.biconnected_components[i].begin(), res.biconnected_components[i].end(),
                    res.biconnected_components[j].begin(), res.biconnected_components[j].end(),
                    std::back_inserter(inter));
                if (inter.size() > 1U) {
                    return false;
                }
                if (inter.size() == 1U &&
                    !std::binary_search(res.articulation_points.begin(),
                                        res.articulation_points.end(), inter[0])) {
                    return false;
                }
            }
        }
    }

    // 4. BridgeBlockTree: complete partition, forest, exact bridge edge bijection
    if (bbt.vertex_to_component.size() != n) {
        return false;
    }
    std::size_t bbt_edges = 0;
    if (!is_undirected_forest(bbt.condensed_tree, &bbt_edges) || bbt_edges != res.bridges.size()) {
        return false;
    }
    for (VertexId u = 0; u < static_cast<VertexId>(n); ++u) {
        const std::size_t cu = bbt.vertex_to_component[u];
        if (cu >= bbt.condensed_tree.num_vertices()) {
            return false;
        }
        for (const VertexId v : graph.out_neighbors(u)) {
            if (u < v) {
                const std::size_t cv = bbt.vertex_to_component[v];
                const bool is_bridge =
                    std::binary_search(res.bridges.begin(), res.bridges.end(),
                                       std::pair<std::size_t, std::size_t>{u, v});
                if ((cu != cv) != is_bridge) {
                    return false;
                }
                if (is_bridge &&
                    !bbt.condensed_tree
                         .find_out_edge_index(static_cast<VertexId>(cu), static_cast<VertexId>(cv))
                         .has_value()) {
                    return false;
                }
            }
        }
    }

    // 5. BlockCutTree: strictly bipartite forest between [0, num_art) and [num_art, num_art +
    // num_blk)
    if (bct.num_articulation_nodes != res.articulation_points.size() ||
        bct.num_block_nodes != res.biconnected_components.size() ||
        bct.articulation_points != res.articulation_points ||
        bct.blocks != res.biconnected_components ||
        bct.tree.num_vertices() != bct.num_articulation_nodes + bct.num_block_nodes ||
        !is_undirected_forest(bct.tree)) {
        return false;
    }

    for (std::size_t cut = 0; cut < bct.num_articulation_nodes; ++cut) {
        const VertexId cut_node = static_cast<VertexId>(cut);
        if (bct.tree.out_degree(cut_node) < 2U) {
            return false;
        }
        for (const VertexId nb : bct.tree.out_neighbors(cut_node)) {
            if (nb < bct.num_articulation_nodes) {
                return false;
            }
            const auto& blk = bct.blocks[nb - bct.num_articulation_nodes];
            if (!std::binary_search(blk.begin(), blk.end(), bct.articulation_points[cut])) {
                return false;
            }
        }
    }
    for (std::size_t b = 0; b < bct.num_block_nodes; ++b) {
        const VertexId blk_node = static_cast<VertexId>(bct.num_articulation_nodes + b);
        for (const VertexId nb : bct.tree.out_neighbors(blk_node)) {
            if (nb >= bct.num_articulation_nodes) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Deterministic diverse random graph generator for differential testing across V in [1,
 * 10^5].
 */
[[nodiscard]] inline std::vector<std::pair<VertexId, VertexId>>
generate_random_topology(std::mt19937_64& rng, std::size_t num_v, uint32_t mode) {
    std::vector<std::pair<VertexId, VertexId>> edges;
    if (num_v == 0) {
        return edges;
    }

    switch (mode % 5U) {
    case 0U: {
        // Erdos-Renyi random graph (including sparse and dense small graphs)
        const std::size_t max_edges = num_v <= 64 ? (num_v * num_v) / 2U + num_v : num_v * 3U;
        const std::size_t m = rng() % (max_edges + 1U);
        edges.reserve(m);
        for (std::size_t i = 0; i < m; ++i) {
            edges.emplace_back(static_cast<VertexId>(rng() % num_v),
                               static_cast<VertexId>(rng() % num_v));
        }
        break;
    }
    case 1U: {
        // Random spanning tree + sparse chords (guarantees mix of bridges and cycles)
        edges.reserve(num_v + num_v / 3U);
        for (std::size_t v = 1; v < num_v; ++v) {
            const VertexId p = static_cast<VertexId>(rng() % v);
            edges.emplace_back(p, static_cast<VertexId>(v));
        }
        const std::size_t chords = (num_v > 1) ? (rng() % std::max<std::size_t>(2, num_v / 2U)) : 0;
        for (std::size_t i = 0; i < chords; ++i) {
            edges.emplace_back(static_cast<VertexId>(rng() % num_v),
                               static_cast<VertexId>(rng() % num_v));
        }
        break;
    }
    case 2U: {
        // Disconnected forest / cactus components with isolated vertices and self-loops
        const std::size_t num_edges = rng() % (num_v * 2U + 1U);
        edges.reserve(num_edges);
        const std::size_t clusters = std::max<std::size_t>(1, std::min<std::size_t>(num_v, 6));
        for (std::size_t i = 0; i < num_edges; ++i) {
            if ((rng() % 11U) == 0U) {
                // Cross-cluster bridge/edge or self-loop
                const VertexId u = static_cast<VertexId>(rng() % num_v);
                const VertexId v = (rng() & 1U) ? u : static_cast<VertexId>(rng() % num_v);
                edges.emplace_back(u, v);
            } else {
                const std::size_t bucket = rng() % clusters;
                const std::size_t lo = (bucket * num_v) / clusters;
                const std::size_t hi = ((bucket + 1U) * num_v) / clusters;
                if (hi > lo) {
                    const VertexId u = static_cast<VertexId>(lo + (rng() % (hi - lo)));
                    const VertexId v = static_cast<VertexId>(lo + (rng() % (hi - lo)));
                    edges.emplace_back(u, v);
                }
            }
        }
        break;
    }
    case 3U: {
        // Chain of cycles (cactus) + bridges
        std::size_t cur = 0;
        while (cur + 1 < num_v) {
            const std::size_t step = 2U + (rng() % 5U);
            const std::size_t end = std::min(num_v - 1U, cur + step);
            for (std::size_t v = cur; v < end; ++v) {
                edges.emplace_back(static_cast<VertexId>(v), static_cast<VertexId>(v + 1U));
            }
            if ((rng() % 3U) != 0U && end > cur + 1U) {
                edges.emplace_back(static_cast<VertexId>(cur), static_cast<VertexId>(end));
            }
            cur = end;
        }
        break;
    }
    default: {
        // Path/barbell permutation with occasional skip chords
        std::vector<VertexId> perm(num_v);
        std::iota(perm.begin(), perm.end(), VertexId{0});
        if (num_v <= 4096U) {
            std::shuffle(perm.begin(), perm.end(), rng);
        }
        for (std::size_t i = 1; i < num_v; ++i) {
            if ((rng() % 10U) != 0U) {
                edges.emplace_back(perm[i - 1U], perm[i]);
            }
        }
        const std::size_t extra = rng() % std::max<std::size_t>(1, num_v / 4U);
        for (std::size_t i = 0; i < extra; ++i) {
            const std::size_t a = rng() % num_v;
            const std::size_t b = rng() % num_v;
            edges.emplace_back(perm[a], perm[b]);
        }
        break;
    }
    }

    return edges;
}

} // namespace algoat::graph::test

/**
 * @file biconnectivity.hpp
 * @brief Hopcroft-Tarjan biconnectivity, articulation points, bridges, and tree condensation
 * engines for static undirected CSR graphs.
 *
 * Implements:
 * 1. Single-pass O(V + E) non-recursive Hopcroft-Tarjan (1973) DFS over static CSRGraph.
 * 2. Depth-bounded 64-byte cache-aligned pre-reserved stacks preventing OS call-stack
 *    overflow on deep linear/path topologies (V >= 10^6) and eliminating 2x vector spikes.
 * 3. Working memory bounded strictly within 3 * V * 8 bytes via packed 32-bit timestamps,
 *    stack-resident lowpoint frames, and an O(V) vertex stack.
 * 4. O(V + E) 2-edge-connected Bridge-Block Tree condensation and bipartite Block-Cut Tree
 *    construction.
 */

#pragma once

#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::graph {

namespace detail {

inline constexpr uint32_t kArticulationBit = uint32_t{1} << 31;
inline constexpr uint32_t kClosedCompBit = uint32_t{1} << 31;
inline constexpr uint32_t kTinMask = kArticulationBit - 1U;
inline constexpr VertexId kInvalidVertex = std::numeric_limits<VertexId>::max();

struct DfsFrame {
    VertexId u;
    uint32_t edge_idx;
    uint32_t low;
};
static_assert(sizeof(DfsFrame) == 12, "DfsFrame must be tightly packed to 12 bytes");

template <typename WeightType>
inline void validate_biconnectivity_graph(const CSRGraph<WeightType>& graph, const char* fn_name) {
    if (graph.is_directed()) {
        throw std::invalid_argument(std::string(fn_name) + ": requires an undirected CSRGraph");
    }
    if (static_cast<std::size_t>(graph.num_vertices()) > static_cast<std::size_t>(kTinMask)) {
        throw std::length_error(std::string(fn_name) +
                                ": num_vertices exceeds maximum indexable timestamp");
    }
}

} // namespace detail

/**
 * @struct BiconnectivityResult
 * @brief Output container holding articulation points, normalized bridges, and biconnected blocks.
 */
struct BiconnectivityResult {
    std::vector<std::size_t> articulation_points;
    std::vector<std::pair<std::size_t, std::size_t>> bridges;
    std::vector<std::vector<std::size_t>> biconnected_components; // Component vertex lists
};

/**
 * @brief Computes all articulation vertices, bridges, and biconnected vertex blocks in O(V + E)
 * time.
 *
 * Employs non-recursive Hopcroft-Tarjan (1973) DFS over contiguous CSR slices with bounded
 * 64-byte cache-aligned paged stacks. Self-loops and isolated vertices do not form biconnected
 * blocks or bridges, matching NetworkX semantics.
 *
 * @tparam WeightType Edge weight type of the CSR graph (`void` for unweighted graphs).
 * @param graph Undirected input CSR graph.
 * @return BiconnectivityResult with sorted articulation points, lexicographically sorted
 * bridges `(min(u, v), max(u, v))`, and sorted biconnected block vertex lists.
 * @throws std::invalid_argument if @p graph is directed.
 * @throws std::length_error if `graph.num_vertices()` exceeds `2^31 - 1`.
 */
template <typename WeightType = void>
[[nodiscard]] inline BiconnectivityResult find_biconnectivity(const CSRGraph<WeightType>& graph) {
    detail::validate_biconnectivity_graph(graph, "find_biconnectivity");

    BiconnectivityResult result;
    const std::size_t num_v = graph.num_vertices();
    if (num_v == 0) {
        return result;
    }

    std::vector<uint32_t, detail::AlignedAllocator<uint32_t, 64>> tin(num_v, 0U);
    std::vector<detail::DfsFrame, detail::AlignedAllocator<detail::DfsFrame, 64>> call_stack;
    call_stack.reserve(num_v);
    std::vector<VertexId, detail::AlignedAllocator<VertexId, 64>> v_stack;
    v_stack.reserve(num_v);

    uint32_t timer = 0;

    for (VertexId root = 0; root < static_cast<VertexId>(num_v); ++root) {
        if ((tin[root] & detail::kTinMask) != 0U) {
            continue;
        }

        tin[root] = ++timer;
        call_stack.push_back(detail::DfsFrame{root, 0U, tin[root]});
        v_stack.push_back(root);
        uint32_t root_children = 0;

        while (!call_stack.empty()) {
            const detail::DfsFrame cur = call_stack.back();
            const VertexId u = cur.u;
            const VertexId parent = (call_stack.size() >= 2U) ? call_stack[call_stack.size() - 2U].u
                                                              : detail::kInvalidVertex;
            const auto neighbors = graph.out_neighbors(u);

            if (cur.edge_idx < neighbors.size()) {
                const VertexId v = neighbors[cur.edge_idx];
                call_stack.back().edge_idx = cur.edge_idx + 1U;
                if (v == u || v == parent) {
                    continue;
                }

                const uint32_t v_tin = tin[v] & detail::kTinMask;
                if (v_tin == 0U) {
                    if (parent == detail::kInvalidVertex) {
                        ++root_children;
                    }
                    const uint32_t child_tin = ++timer;
                    tin[v] = child_tin;
                    call_stack.push_back(detail::DfsFrame{v, 0U, child_tin});
                    v_stack.push_back(v);
                } else if (v_tin < cur.low) {
                    call_stack.back().low = v_tin;
                }
            } else {
                const uint32_t u_low = cur.low;
                call_stack.pop_back();

                if (!call_stack.empty()) {
                    detail::DfsFrame& parent_frame = call_stack.back();
                    const VertexId p = parent_frame.u;
                    const uint32_t p_tin = tin[p] & detail::kTinMask;
                    parent_frame.low = std::min(parent_frame.low, u_low);

                    if (u_low > p_tin) {
                        result.bridges.emplace_back(std::min<std::size_t>(p, u),
                                                    std::max<std::size_t>(p, u));
                    }
                    if (u_low >= p_tin) {
                        if (call_stack.size() > 1U) {
                            tin[p] |= detail::kArticulationBit;
                        }
                        std::vector<std::size_t> comp;
                        comp.push_back(static_cast<std::size_t>(p));
                        while (!v_stack.empty()) {
                            const VertexId w = v_stack.back();
                            v_stack.pop_back();
                            comp.push_back(static_cast<std::size_t>(w));
                            if (w == u) {
                                break;
                            }
                        }
                        std::sort(comp.begin(), comp.end());
                        result.biconnected_components.push_back(std::move(comp));
                    }
                } else {
                    v_stack.pop_back();
                    if (root_children >= 2U) {
                        tin[u] |= detail::kArticulationBit;
                    }
                }
            }
        }
    }

    for (std::size_t u = 0; u < num_v; ++u) {
        if ((tin[u] & detail::kArticulationBit) != 0U) {
            result.articulation_points.push_back(u);
        }
    }
    std::sort(result.bridges.begin(), result.bridges.end());

    return result;
}

/**
 * @struct BridgeBlockTree
 * @brief Condensed 2-edge-connected component forest where each edge corresponds to a bridge.
 */
struct BridgeBlockTree {
    CSRGraph<void> condensed_tree;
    std::vector<std::size_t> vertex_to_component;
};

/**
 * @brief Contracts each 2-edge-connected component into a single node and links components via
 * bridges in O(V + E) time.
 *
 * Uses Nuutila-style post-order follower tracking and overwrites closed entries in `tin` once a
 * 2-edge-connected component seals, keeping auxiliary scratch memory strictly bounded by
 * 3 * V * 8 bytes.
 *
 * @tparam WeightType Edge weight type of the CSR graph (`void` for unweighted graphs).
 * @param graph Undirected input CSR graph.
 * @return BridgeBlockTree containing the condensed undirected CSR forest and total vertex
 * partition `vertex_to_component`.
 * @throws std::invalid_argument if @p graph is directed.
 * @throws std::length_error if `graph.num_vertices()` exceeds `2^31 - 1`.
 */
template <typename WeightType = void>
[[nodiscard]] inline BridgeBlockTree build_bridge_block_tree(const CSRGraph<WeightType>& graph) {
    detail::validate_biconnectivity_graph(graph, "build_bridge_block_tree");

    BridgeBlockTree result;
    const std::size_t num_v = graph.num_vertices();
    if (num_v == 0) {
        result.condensed_tree = CSRGraph<void>::build_from_edges(0, {}, /*directed=*/false);
        return result;
    }

    std::vector<std::pair<VertexId, VertexId>> tree_edges;
    uint32_t num_ecc = 0;

    {
        std::vector<uint32_t, detail::AlignedAllocator<uint32_t, 64>> tin(num_v, 0U);
        std::size_t num_bridges = 0;

        {
            std::vector<detail::DfsFrame, detail::AlignedAllocator<detail::DfsFrame, 64>>
                call_stack;
            call_stack.reserve(num_v);
            std::vector<VertexId, detail::AlignedAllocator<VertexId, 64>> ecc_stack;
            ecc_stack.reserve(num_v);
            uint32_t timer = 0;

            for (VertexId root = 0; root < static_cast<VertexId>(num_v); ++root) {
                if (tin[root] != 0U) {
                    continue;
                }

                tin[root] = ++timer;
                call_stack.push_back(detail::DfsFrame{root, 0U, tin[root]});

                while (!call_stack.empty()) {
                    const detail::DfsFrame cur = call_stack.back();
                    const VertexId u = cur.u;
                    const VertexId parent = (call_stack.size() >= 2U)
                                                ? call_stack[call_stack.size() - 2U].u
                                                : detail::kInvalidVertex;
                    const auto neighbors = graph.out_neighbors(u);

                    if (cur.edge_idx < neighbors.size()) {
                        const VertexId v = neighbors[cur.edge_idx];
                        call_stack.back().edge_idx = cur.edge_idx + 1U;
                        if (v == u || v == parent) {
                            continue;
                        }

                        const uint32_t v_raw = tin[v];
                        if (v_raw == 0U) {
                            const uint32_t child_tin = ++timer;
                            tin[v] = child_tin;
                            call_stack.push_back(detail::DfsFrame{v, 0U, child_tin});
                        } else if ((v_raw & detail::kClosedCompBit) == 0U && v_raw < cur.low) {
                            call_stack.back().low = v_raw;
                        }
                    } else {
                        const uint32_t u_low = cur.low;
                        const uint32_t u_tin = tin[u];
                        call_stack.pop_back();

                        if (!call_stack.empty()) {
                            detail::DfsFrame& parent_frame = call_stack.back();
                            parent_frame.low = std::min(parent_frame.low, u_low);
                        }

                        if (u_low == u_tin) {
                            if (!call_stack.empty()) {
                                ++num_bridges;
                            }
                            const uint32_t comp_id = num_ecc++;
                            tin[u] = detail::kClosedCompBit | comp_id;
                            while (!ecc_stack.empty()) {
                                const VertexId w = ecc_stack.back();
                                if (tin[w] <= u_tin) {
                                    break;
                                }
                                ecc_stack.pop_back();
                                tin[w] = detail::kClosedCompBit | comp_id;
                            }
                        } else {
                            ecc_stack.push_back(u);
                        }
                    }
                }
            }
        }

        tree_edges.reserve(num_bridges);
        for (VertexId u = 0; u < static_cast<VertexId>(num_v); ++u) {
            const VertexId comp_u = static_cast<VertexId>(tin[u] & detail::kTinMask);
            for (const VertexId v : graph.out_neighbors(u)) {
                if (u < v) {
                    const VertexId comp_v = static_cast<VertexId>(tin[v] & detail::kTinMask);
                    if (comp_u != comp_v) {
                        tree_edges.emplace_back(comp_u, comp_v);
                    }
                }
            }
        }

        result.vertex_to_component.resize(num_v);
        for (std::size_t u = 0; u < num_v; ++u) {
            result.vertex_to_component[u] = static_cast<std::size_t>(tin[u] & detail::kTinMask);
        }
    }

    result.condensed_tree =
        CSRGraph<void>::build_from_edges(num_ecc, tree_edges, /*directed=*/false);
    return result;
}

/**
 * @struct BlockCutTree
 * @brief Bipartite Block-Cut forest connecting articulation vertices [0, num_articulation_nodes)
 * with biconnected blocks [num_articulation_nodes, num_articulation_nodes + num_block_nodes).
 */
struct BlockCutTree {
    CSRGraph<void> tree;
    std::vector<std::size_t> articulation_points;
    std::vector<std::vector<std::size_t>> blocks;
    std::size_t num_articulation_nodes{0};
    std::size_t num_block_nodes{0};
};

/**
 * @brief Constructs the bipartite Block-Cut Tree / Forest in O(V + E) time.
 *
 * @tparam WeightType Edge weight type of the CSR graph (`void` for unweighted graphs).
 * @param graph Undirected input CSR graph.
 * @return BlockCutTree with undirected bipartite CSR forest and node metadata.
 * @throws std::invalid_argument if @p graph is directed.
 * @throws std::length_error if `graph.num_vertices()` exceeds `2^31 - 1`.
 */
template <typename WeightType = void>
[[nodiscard]] inline BlockCutTree build_block_cut_tree(const CSRGraph<WeightType>& graph) {
    detail::validate_biconnectivity_graph(graph, "build_block_cut_tree");

    BlockCutTree result;
    auto bicon = find_biconnectivity(graph);
    result.articulation_points = std::move(bicon.articulation_points);
    result.blocks = std::move(bicon.biconnected_components);
    result.num_articulation_nodes = result.articulation_points.size();
    result.num_block_nodes = result.blocks.size();

    const std::size_t total_nodes = result.num_articulation_nodes + result.num_block_nodes;
    if (total_nodes == 0) {
        result.tree = CSRGraph<void>::build_from_edges(0, {}, /*directed=*/false);
        return result;
    }

    std::vector<std::pair<VertexId, VertexId>> tree_edges;
    {
        std::vector<VertexId, detail::AlignedAllocator<VertexId, 64>> cut_index(
            graph.num_vertices(), detail::kInvalidVertex);
        for (std::size_t i = 0; i < result.num_articulation_nodes; ++i) {
            cut_index[result.articulation_points[i]] = static_cast<VertexId>(i);
        }

        for (std::size_t b = 0; b < result.num_block_nodes; ++b) {
            const VertexId block_node = static_cast<VertexId>(result.num_articulation_nodes + b);
            for (const std::size_t v : result.blocks[b]) {
                const VertexId cut_node = cut_index[v];
                if (cut_node != detail::kInvalidVertex) {
                    tree_edges.emplace_back(cut_node, block_node);
                }
            }
        }
    }

    result.tree = CSRGraph<void>::build_from_edges(total_nodes, tree_edges, /*directed=*/false);
    return result;
}

} // namespace algoat::graph

/**
 * @file biconnectivity.hpp
 * @brief Hopcroft-Tarjan biconnectivity, articulation points, bridges, and tree condensation
 * engines for static undirected CSR graphs.
 *
 * Implements:
 * 1. Single-pass O(V + E) non-recursive Hopcroft-Tarjan (1973) DFS over static CSRGraph.
 * 2. Depth-bounded 64-byte cache-aligned paged small-vector stacks preventing OS call-stack
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

/**
 * @class CacheAlignedPagedStack
 * @brief Depth-bounded, 64-byte cache-aligned paged small-vector stack avoiding 2x geometric
 * reallocation spikes and OS call-stack overflow on deep linear graphs.
 */
template <typename T, std::size_t InlineCount, std::size_t PageElems>
class alignas(64) CacheAlignedPagedStack {
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);
    static_assert(InlineCount > 0 && PageElems > 0);
    static_assert((PageElems & (PageElems - 1U)) == 0U, "PageElems must be a power of two");

public:
    explicit CacheAlignedPagedStack(std::size_t max_capacity) noexcept
        : inline_buf_{}, pages_{}, alloc_{}, size_(0), allocated_capacity_(InlineCount),
          max_capacity_(max_capacity) {}

    CacheAlignedPagedStack(const CacheAlignedPagedStack&) = delete;
    CacheAlignedPagedStack& operator=(const CacheAlignedPagedStack&) = delete;
    CacheAlignedPagedStack(CacheAlignedPagedStack&&) = delete;
    CacheAlignedPagedStack& operator=(CacheAlignedPagedStack&&) = delete;

    ~CacheAlignedPagedStack() {
        for (const Page& page : pages_) {
            alloc_.deallocate(page.ptr, page.count);
        }
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    void push(const T& value) {
        if (size_ == allocated_capacity_) {
            allocate_next_page();
        }
        at(size_) = value;
        ++size_;
    }

    void pop() noexcept {
        if (size_ > 0) {
            --size_;
        }
    }

    [[nodiscard]] T& top() noexcept {
        return at(size_ - 1U);
    }

    [[nodiscard]] const T& top() const noexcept {
        return at(size_ - 1U);
    }

    [[nodiscard]] T& at(std::size_t idx) noexcept {
        if (idx < InlineCount) {
            return inline_buf_[idx];
        }
        const std::size_t overflow = idx - InlineCount;
        return pages_[overflow / PageElems].ptr[overflow & (PageElems - 1U)];
    }

    [[nodiscard]] const T& at(std::size_t idx) const noexcept {
        if (idx < InlineCount) {
            return inline_buf_[idx];
        }
        const std::size_t overflow = idx - InlineCount;
        return pages_[overflow / PageElems].ptr[overflow & (PageElems - 1U)];
    }

private:
    struct Page {
        T* ptr;
        std::size_t count;
    };

    void allocate_next_page() {
        if (allocated_capacity_ >= max_capacity_) {
            throw std::length_error("CacheAlignedPagedStack: capacity exceeded");
        }
        if (pages_.empty() && max_capacity_ > InlineCount) {
            const std::size_t max_pages =
                (max_capacity_ - InlineCount + PageElems - 1U) / PageElems;
            pages_.reserve(max_pages);
        }
        const std::size_t remaining = max_capacity_ - allocated_capacity_;
        const std::size_t chunk = std::min(PageElems, remaining);
        T* mem = alloc_.allocate(chunk);
        try {
            pages_.push_back(Page{mem, chunk});
        } catch (...) {
            alloc_.deallocate(mem, chunk);
            throw;
        }
        allocated_capacity_ += chunk;
    }

    std::array<T, InlineCount> inline_buf_;
    std::vector<Page, AlignedAllocator<Page, 64>> pages_;
    AlignedAllocator<T, 64> alloc_;
    std::size_t size_;
    std::size_t allocated_capacity_;
    std::size_t max_capacity_;
};

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
    detail::CacheAlignedPagedStack<detail::DfsFrame, 256, 4096> call_stack(num_v);
    detail::CacheAlignedPagedStack<VertexId, 512, 4096> v_stack(num_v);

    uint32_t timer = 0;

    for (VertexId root = 0; root < static_cast<VertexId>(num_v); ++root) {
        if ((tin[root] & detail::kTinMask) != 0U) {
            continue;
        }

        tin[root] = ++timer;
        call_stack.push(detail::DfsFrame{root, 0U, tin[root]});
        v_stack.push(root);
        uint32_t root_children = 0;

        while (!call_stack.empty()) {
            detail::DfsFrame& cur = call_stack.top();
            const VertexId u = cur.u;
            const VertexId parent = (call_stack.size() >= 2U)
                                        ? call_stack.at(call_stack.size() - 2U).u
                                        : detail::kInvalidVertex;
            const auto neighbors = graph.out_neighbors(u);

            if (cur.edge_idx < neighbors.size()) {
                const VertexId v = neighbors[cur.edge_idx++];
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
                    call_stack.push(detail::DfsFrame{v, 0U, child_tin});
                    v_stack.push(v);
                } else {
                    cur.low = std::min(cur.low, v_tin);
                }
            } else {
                const uint32_t u_low = cur.low;
                call_stack.pop();

                if (!call_stack.empty()) {
                    detail::DfsFrame& parent_frame = call_stack.top();
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
                            const VertexId w = v_stack.top();
                            v_stack.pop();
                            comp.push_back(static_cast<std::size_t>(w));
                            if (w == u) {
                                break;
                            }
                        }
                        std::sort(comp.begin(), comp.end());
                        result.biconnected_components.push_back(std::move(comp));
                    }
                } else {
                    v_stack.pop();
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
            detail::CacheAlignedPagedStack<detail::DfsFrame, 256, 4096> call_stack(num_v);
            detail::CacheAlignedPagedStack<VertexId, 512, 4096> ecc_stack(num_v);
            uint32_t timer = 0;

            for (VertexId root = 0; root < static_cast<VertexId>(num_v); ++root) {
                if (tin[root] != 0U) {
                    continue;
                }

                tin[root] = ++timer;
                call_stack.push(detail::DfsFrame{root, 0U, tin[root]});

                while (!call_stack.empty()) {
                    detail::DfsFrame& cur = call_stack.top();
                    const VertexId u = cur.u;
                    const VertexId parent = (call_stack.size() >= 2U)
                                                ? call_stack.at(call_stack.size() - 2U).u
                                                : detail::kInvalidVertex;
                    const auto neighbors = graph.out_neighbors(u);

                    if (cur.edge_idx < neighbors.size()) {
                        const VertexId v = neighbors[cur.edge_idx++];
                        if (v == u || v == parent) {
                            continue;
                        }

                        const uint32_t v_raw = tin[v];
                        if (v_raw == 0U) {
                            const uint32_t child_tin = ++timer;
                            tin[v] = child_tin;
                            call_stack.push(detail::DfsFrame{v, 0U, child_tin});
                        } else if ((v_raw & detail::kClosedCompBit) == 0U) {
                            cur.low = std::min(cur.low, v_raw);
                        }
                    } else {
                        const uint32_t u_low = cur.low;
                        const uint32_t u_tin = tin[u];
                        call_stack.pop();

                        if (!call_stack.empty()) {
                            detail::DfsFrame& parent_frame = call_stack.top();
                            parent_frame.low = std::min(parent_frame.low, u_low);
                        }

                        if (u_low == u_tin) {
                            if (!call_stack.empty()) {
                                ++num_bridges;
                            }
                            const uint32_t comp_id = num_ecc++;
                            tin[u] = detail::kClosedCompBit | comp_id;
                            while (!ecc_stack.empty()) {
                                const VertexId w = ecc_stack.top();
                                if (tin[w] <= u_tin) {
                                    break;
                                }
                                ecc_stack.pop();
                                tin[w] = detail::kClosedCompBit | comp_id;
                            }
                        } else {
                            ecc_stack.push(u);
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

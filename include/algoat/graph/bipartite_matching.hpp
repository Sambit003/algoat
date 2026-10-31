/**
 * @file bipartite_matching.hpp
 * @brief High-performance Hopcroft-Karp Maximum Cardinality Bipartite Matching in O(E sqrt(V)).
 *
 * Implements:
 * 1. Hopcroft-Karp Algorithm (1973) operating in strictly O(E * sqrt(V)) worst-case time:
 *    - Fast greedy initialization pass (Duff, Kaya, & Ucar 2011) resolving the majority of
 *      matchings in a single linear pass.
 *    - Multi-source Breadth-First Search (BFS) constructing layered distance graphs using a flat
 *      ring-buffer queue over the CSR adjacency representation.
 *    - Multi-source Depth-First Search (DFS) with persistent current-arc pointers and dead-end
 *      pruning extracting a maximal set of vertex-disjoint augmenting paths per phase.
 *    - Flat Compressed Sparse Row (CSR) adjacency storage eliminating pointer chasing across
 *      dynamically allocated inner vectors.
 * 2. Minimum Vertex Cover construction via Konig's Theorem duality in O(V + E) time.
 * 3. Maximum Independent Set via vertex cover complementation in O(V + E) time.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace algoat::graph {

using VertexId = uint32_t;

/**
 * @struct BipartiteMatchingResult
 * @brief Output container for maximum cardinality bipartite matching.
 */
struct BipartiteMatchingResult {
    std::size_t matching_size{0};
    std::vector<VertexId> match_left;  // Size: |L|, maps u in L -> matched v in R (or kUnmatched)
    std::vector<VertexId> match_right; // Size: |R|, maps v in R -> matched u in L (or kUnmatched)
};

/**
 * @class BipartiteGraph
 * @brief High-performance undirected bipartite graph and matching engine.
 *
 * Edges are added incrementally via add_edge(). On the first call to maximum_matching() (or any
 * duality query), the builder adjacency list is compacted into a flat CSR representation that
 * eliminates pointer chasing during the BFS and DFS phases.
 */
class BipartiteGraph {
public:
    static constexpr VertexId kUnmatched = std::numeric_limits<VertexId>::max();

    /**
     * @brief Constructs a bipartite graph with specified partition sizes.
     * @param num_left  Number of vertices in the left partition L (indices [0, num_left - 1]).
     * @param num_right Number of vertices in the right partition R (indices [0, num_right - 1]).
     */
    BipartiteGraph(std::size_t num_left, std::size_t num_right)
        : num_left_(num_left), num_right_(num_right), builder_adj_(num_left) {}

    /**
     * @brief Adds an undirected edge between left vertex u and right vertex v.
     * @throws std::out_of_range if u >= num_left or v >= num_right.
     */
    void add_edge(VertexId u, VertexId v) {
        if (u >= num_left_ || v >= num_right_) {
            throw std::out_of_range("Vertex index out of bounds in BipartiteGraph::add_edge");
        }
        builder_adj_[u].push_back(v);
        ++num_edges_;
        csr_stale_ = true;
    }

    /**
     * @brief Computes Maximum Cardinality Bipartite Matching in strictly O(E * sqrt(V)) time.
     *
     * Combines greedy pre-matching (SGM), multi-source BFS level graph construction, and
     * multi-source DFS with current-arc pointer pruning. All scratch buffers are allocated once
     * before the phasing loop and reused across iterations.
     */
    [[nodiscard]] BipartiteMatchingResult maximum_matching() const {
        build_csr();

        BipartiteMatchingResult result;
        result.matching_size = 0;
        result.match_left.assign(num_left_, kUnmatched);
        result.match_right.assign(num_right_, kUnmatched);

        if (num_left_ == 0 || num_right_ == 0 || num_edges_ == 0) {
            return result;
        }

        // Fast greedy pre-matching pass (SGM heuristic: Duff, Kaya, & Ucar 2011)
        for (VertexId u = 0; u < num_left_; ++u) {
            const uint32_t start = head_[u];
            const uint32_t end = head_[u + 1];
            for (uint32_t i = start; i < end; ++i) {
                const VertexId v = to_[i];
                if (result.match_right[v] == kUnmatched) {
                    result.match_left[u] = v;
                    result.match_right[v] = u;
                    ++result.matching_size;
                    break;
                }
            }
        }

        // kNilVertex is the sentinel representing "unmatched" in the pair_right array.
        // It occupies slot num_left_ in the dist array (size num_left_ + 1).
        const VertexId kNilVertex = static_cast<VertexId>(num_left_);

        // Scratch buffers allocated once for all Hopcroft-Karp phases
        std::vector<uint32_t> dist(num_left_ + 1, kDistInf);
        std::vector<uint32_t> ptr(num_left_, 0);   // Current-arc pointers (CSR edge indices)
        std::vector<VertexId> queue(num_left_, 0); // Flat BFS ring buffer

        // pair_right[v] = left vertex matched to v, or kNilVertex if v is unmatched
        std::vector<VertexId> pair_right(num_right_);
        for (VertexId v = 0; v < num_right_; ++v) {
            pair_right[v] =
                (result.match_right[v] == kUnmatched) ? kNilVertex : result.match_right[v];
        }

        // Hopcroft-Karp phasing loop: terminates in at most 2 * sqrt(V) iterations
        while (bfs_layer(result.match_left, pair_right, dist, queue, kNilVertex)) {
            // Reset current-arc pointers to CSR row start for each left vertex
            for (VertexId u = 0; u < num_left_; ++u) {
                ptr[u] = head_[u];
            }

            for (VertexId u = 0; u < num_left_; ++u) {
                if (result.match_left[u] == kUnmatched &&
                    dfs_augment(u, result.match_left, pair_right, dist, ptr, kNilVertex)) {
                    ++result.matching_size;
                }
            }
        }

        // Propagate right matches back into the result
        for (VertexId v = 0; v < num_right_; ++v) {
            result.match_right[v] = (pair_right[v] == kNilVertex) ? kUnmatched : pair_right[v];
        }

        return result;
    }

    /**
     * @brief Computes Minimum Vertex Cover via Konig's Theorem duality in O(V + E) time.
     *
     * Finds the reachability set Z from unmatched left vertices via alternating paths.
     * Minimum vertex cover is (L \ Z) U (R ∩ Z).
     * Guarantees: |Cover| == |Matching|, and every edge has at least one endpoint in Cover.
     */
    [[nodiscard]] std::pair<std::vector<VertexId>, std::vector<VertexId>>
    minimum_vertex_cover(const BipartiteMatchingResult& matching) const {
        if (matching.match_left.size() != num_left_ || matching.match_right.size() != num_right_) {
            throw std::invalid_argument("Matching dimensions mismatch in minimum_vertex_cover");
        }

        if (num_left_ == 0 || num_right_ == 0 || matching.matching_size == 0) {
            return {{}, {}};
        }

        build_csr();

        std::vector<bool> visited_l(num_left_, false);
        std::vector<bool> visited_r(num_right_, false);
        alternating_bfs_reachability(matching, visited_l, visited_r);

        std::vector<VertexId> cover_l;
        std::vector<VertexId> cover_r;
        cover_l.reserve(num_left_);
        cover_r.reserve(num_right_);

        for (VertexId u = 0; u < num_left_; ++u) {
            if (!visited_l[u]) {
                cover_l.push_back(u);
            }
        }
        for (VertexId v = 0; v < num_right_; ++v) {
            if (visited_r[v]) {
                cover_r.push_back(v);
            }
        }

        return {std::move(cover_l), std::move(cover_r)};
    }

    /**
     * @brief Computes Maximum Independent Set via vertex cover complementation in O(V + E) time.
     *
     * Maximum independent set is V \ (Minimum Vertex Cover).
     * Guarantees: No two vertices in the set share an edge, |Indep| == |V| - |Matching|.
     */
    [[nodiscard]] std::pair<std::vector<VertexId>, std::vector<VertexId>>
    maximum_independent_set(const BipartiteMatchingResult& matching) const {
        const auto [cover_l, cover_r] = minimum_vertex_cover(matching);

        std::vector<bool> in_cover_l(num_left_, false);
        for (VertexId u : cover_l) {
            in_cover_l[u] = true;
        }

        std::vector<bool> in_cover_r(num_right_, false);
        for (VertexId v : cover_r) {
            in_cover_r[v] = true;
        }

        std::vector<VertexId> indep_l;
        std::vector<VertexId> indep_r;
        indep_l.reserve(num_left_ - cover_l.size());
        indep_r.reserve(num_right_ - cover_r.size());

        for (VertexId u = 0; u < num_left_; ++u) {
            if (!in_cover_l[u]) {
                indep_l.push_back(u);
            }
        }
        for (VertexId v = 0; v < num_right_; ++v) {
            if (!in_cover_r[v]) {
                indep_r.push_back(v);
            }
        }

        return {std::move(indep_l), std::move(indep_r)};
    }

    [[nodiscard]] std::size_t num_left() const noexcept {
        return num_left_;
    }
    [[nodiscard]] std::size_t num_right() const noexcept {
        return num_right_;
    }
    [[nodiscard]] std::size_t num_edges() const noexcept {
        return num_edges_;
    }

    /**
     * @brief Returns the adjacency list of left vertex u as a contiguous span.
     *
     * Triggers CSR construction on first call after any add_edge().
     * @throws std::out_of_range if u >= num_left().
     */
    [[nodiscard]] std::span<const VertexId> adj(VertexId u) const {
        if (u >= num_left_) {
            throw std::out_of_range("Vertex index out of bounds in BipartiteGraph::adj");
        }
        build_csr();
        return {to_.data() + head_[u], to_.data() + head_[u + 1]};
    }

private:
    static constexpr uint32_t kDistInf = std::numeric_limits<uint32_t>::max();

    std::size_t num_left_{0};
    std::size_t num_right_{0};
    std::size_t num_edges_{0};

    // Builder adjacency list used before first CSR compaction
    std::vector<std::vector<VertexId>> builder_adj_;

    // Flat CSR adjacency: head_[u]..head_[u+1] indexes to_[*]
    mutable bool csr_stale_{false};
    mutable std::vector<uint32_t> head_;
    mutable std::vector<VertexId> to_;

    /**
     * @brief Compacts builder_adj_ into flat CSR arrays (head_, to_) on demand.
     *
     * Idempotent: no-op when CSR is current.
     */
    void build_csr() const {
        if (!csr_stale_ && !head_.empty()) {
            return;
        }
        head_.assign(num_left_ + 1, 0);
        to_.resize(num_edges_);
        for (VertexId u = 0; u < num_left_; ++u) {
            head_[u + 1] = head_[u] + static_cast<uint32_t>(builder_adj_[u].size());
        }
        for (VertexId u = 0; u < num_left_; ++u) {
            std::copy(builder_adj_[u].begin(), builder_adj_[u].end(), to_.begin() + head_[u]);
        }
        csr_stale_ = false;
    }

    /**
     * @brief Alternating BFS reachability from all unmatched left vertices.
     *
     * Shared private helper for minimum_vertex_cover and verify_berge_invariant_internal.
     * On return, visited_l[u]/visited_r[v] indicate reachable vertices.
     */
    void alternating_bfs_reachability(const BipartiteMatchingResult& matching,
                                      std::vector<bool>& visited_l,
                                      std::vector<bool>& visited_r) const {
        std::vector<VertexId> queue(num_left_);
        std::size_t head = 0;
        std::size_t tail = 0;

        for (VertexId u = 0; u < num_left_; ++u) {
            if (matching.match_left[u] == kUnmatched) {
                visited_l[u] = true;
                queue[tail++] = u;
            }
        }

        while (head < tail) {
            const VertexId u = queue[head++];
            const uint32_t start = head_[u];
            const uint32_t end = head_[u + 1];
            for (uint32_t i = start; i < end; ++i) {
                const VertexId v = to_[i];
                // Skip the matched edge from u (traverse only unmatched edges L->R)
                if (v == matching.match_left[u]) {
                    continue;
                }
                if (!visited_r[v]) {
                    visited_r[v] = true;
                    const VertexId matched_u = matching.match_right[v];
                    if (matched_u != kUnmatched && !visited_l[matched_u]) {
                        visited_l[matched_u] = true;
                        queue[tail++] = matched_u;
                    }
                }
            }
        }
    }

    /**
     * @brief Verifies Berge's Lemma: a matching is maximum iff no augmenting paths exist.
     *
     * Called internally by tests via the adj() accessor. Returns true when no augmenting path
     * is found in the residual graph.
     */
    [[nodiscard]] bool
    verify_berge_invariant_internal(const BipartiteMatchingResult& matching) const {
        if (matching.match_left.size() != num_left_ || matching.match_right.size() != num_right_) {
            return false;
        }
        if (num_left_ == 0 || num_right_ == 0) {
            return true;
        }

        build_csr();

        std::vector<bool> visited_l(num_left_, false);
        std::vector<bool> visited_r(num_right_, false);

        std::vector<VertexId> queue(num_left_);
        std::size_t head = 0;
        std::size_t tail = 0;

        for (VertexId u = 0; u < num_left_; ++u) {
            if (matching.match_left[u] == kUnmatched) {
                visited_l[u] = true;
                queue[tail++] = u;
            }
        }

        while (head < tail) {
            const VertexId u = queue[head++];
            const uint32_t start = head_[u];
            const uint32_t end = head_[u + 1];
            for (uint32_t i = start; i < end; ++i) {
                const VertexId v = to_[i];
                if (v == matching.match_left[u]) {
                    continue;
                }
                if (!visited_r[v]) {
                    visited_r[v] = true;
                    // Unmatched right vertex reached: augmenting path exists
                    if (matching.match_right[v] == kUnmatched) {
                        return false;
                    }
                    const VertexId next_u = matching.match_right[v];
                    if (!visited_l[next_u]) {
                        visited_l[next_u] = true;
                        queue[tail++] = next_u;
                    }
                }
            }
        }

        return true;
    }

    /**
     * @brief Phase 1: Multi-source BFS to build the layered level graph.
     *
     * @return true if at least one augmenting path exists (kNilVertex is reachable).
     */
    bool bfs_layer(const std::vector<VertexId>& match_left, const std::vector<VertexId>& pair_right,
                   std::vector<uint32_t>& dist, std::vector<VertexId>& queue,
                   VertexId kNilVertex) const {
        std::size_t head = 0;
        std::size_t tail = 0;

        for (VertexId u = 0; u < num_left_; ++u) {
            if (match_left[u] == kUnmatched) {
                dist[u] = 0;
                queue[tail++] = u;
            } else {
                dist[u] = kDistInf;
            }
        }
        dist[kNilVertex] = kDistInf;

        while (head < tail) {
            const VertexId u = queue[head++];
            if (dist[u] < dist[kNilVertex]) {
                const uint32_t start = head_[u];
                const uint32_t end = head_[u + 1];
                for (uint32_t i = start; i < end; ++i) {
                    const VertexId v = to_[i];
                    const VertexId next_u = pair_right[v];
                    if (dist[next_u] == kDistInf) {
                        dist[next_u] = dist[u] + 1;
                        if (next_u != kNilVertex) {
                            queue[tail++] = next_u;
                        }
                    }
                }
            }
        }

        return dist[kNilVertex] != kDistInf;
    }

    /**
     * @brief Phase 2: DFS to find a single augmenting path from u.
     *
     * On success: rewires pair_right and match_left along the path, invalidates u's level graph
     * entry (dist[u] = kDistInf) to prevent revisiting in the same phase.
     * On dead-end: invalidates u (dist[u] = kDistInf) to prune future DFS calls.
     *
     * @return true if an augmenting path from u to kNilVertex was found and applied.
     */
    bool dfs_augment(VertexId u, std::vector<VertexId>& match_left,
                     std::vector<VertexId>& pair_right, std::vector<uint32_t>& dist,
                     std::vector<uint32_t>& ptr, VertexId kNilVertex) const {
        const uint32_t end = head_[u + 1];
        for (uint32_t& i = ptr[u]; i < end; ++i) {
            const VertexId v = to_[i];
            const VertexId next_u = pair_right[v];

            if (dist[next_u] == dist[u] + 1) {
                const bool found =
                    (next_u == kNilVertex) ||
                    dfs_augment(next_u, match_left, pair_right, dist, ptr, kNilVertex);
                if (found) {
                    pair_right[v] = u;
                    match_left[u] = v;
                    dist[u] = kDistInf; // Mark u as permanently explored for this phase
                    return true;
                }
            }
        }

        // Dead-end: invalidate u from the level graph
        dist[u] = kDistInf;
        return false;
    }

    // Grant test and fuzzer infrastructure access to private verification helper
    friend class BipartiteMatchingTestAccess;
};

/**
 * @brief Test-only friend accessor for Berge invariant verification.
 *
 * Keeps verify_berge_invariant_internal() out of the public API while allowing
 * test and fuzzer code to call it without friendship on every caller.
 */
class BipartiteMatchingTestAccess {
public:
    static bool verify_berge_invariant(const BipartiteGraph& g, const BipartiteMatchingResult& m) {
        return g.verify_berge_invariant_internal(m);
    }
};

} // namespace algoat::graph

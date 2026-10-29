/**
 * @file shortest_path.hpp
 * @brief High-performance Single-Source Shortest Path (SSSP) and point-to-point ALT search engine.
 *
 * Combines cache-aligned 4-ary indexed min-heaps, Dijkstra's algorithm with early termination,
 * Goldberg-Harrelson (2005) ALT (A*, Landmarks, Triangle inequality) goal-directed search,
 * and bidirectional shortest path search.
 */

#pragma once

#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/d_ary_heap.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace algoat::graph {

/**
 * @struct ShortestPathResult
 * @brief Output container for full Single-Source Shortest Path (SSSP) tree exploration.
 */
template <typename WeightType = double> struct ShortestPathResult {
    std::vector<WeightType> distances;
    std::vector<VertexId> predecessors;
    std::size_t nodes_settled{0};
};

/**
 * @class ShortestPathEngine
 * @brief High-performance routing engine for weighted CSR graphs.
 *
 * Implements:
 * 1. Single-source Dijkstra with 4-ary indexed min-heap.
 * 2. Point-to-point Dijkstra with early termination.
 * 3. Bidirectional Dijkstra search.
 * 4. Goldberg-Harrelson ALT search with precomputed landmark distance bounds.
 * 5. Bidirectional ALT search.
 */
template <typename WeightType = double> class ShortestPathEngine {
public:
    static constexpr VertexId NO_PREDECESSOR = std::numeric_limits<VertexId>::max();
    static constexpr WeightType INF_WEIGHT = std::numeric_limits<WeightType>::infinity();

    explicit ShortestPathEngine(const CSRGraph<WeightType>& graph) : graph_(graph) {}

    /**
     * @brief Computes Single-Source Shortest Paths (SSSP) from a given source node.
     *
     * @param source Origin vertex index.
     * @return ShortestPathResult containing all shortest distances, predecessors, and settled
     * count.
     */
    [[nodiscard]] ShortestPathResult<WeightType> dijkstra(VertexId source) const {
        const std::size_t num_v = graph_.num_vertices();
        ShortestPathResult<WeightType> res;
        res.distances.assign(num_v, INF_WEIGHT);
        res.predecessors.assign(num_v, NO_PREDECESSOR);
        res.nodes_settled = 0;

        if (source >= num_v) {
            return res;
        }

        res.distances[source] = WeightType{0};

        d_ary_heap<4, WeightType, VertexId> heap(num_v);
        heap.push(source, WeightType{0});

        while (!heap.empty()) {
            auto [u, d_u] = heap.pop();
            ++res.nodes_settled;

            if (d_u > res.distances[u]) {
                continue;
            }

            auto neighbors = graph_.out_neighbors(u);
            auto weights = graph_.out_weights(u);
            const std::size_t deg = neighbors.size();

            for (std::size_t i = 0; i < deg; ++i) {
                const VertexId v = neighbors[i];
                const WeightType w = weights[i];
                const WeightType new_dist = d_u + w;

                if (new_dist < res.distances[v]) {
                    res.distances[v] = new_dist;
                    res.predecessors[v] = u;
                    heap.push_or_decrease_key(v, new_dist);
                }
            }
        }

        return res;
    }

    /**
     * @brief Point-to-point Dijkstra search with early termination when target is settled.
     *
     * @param source Origin vertex index.
     * @param target Destination vertex index.
     * @param path_out Optional output vector to store reconstructed path vertices [source ...
     * target].
     * @return Shortest distance if reachable, or std::nullopt.
     */
    [[nodiscard]] std::optional<WeightType>
    dijkstra_p2p(VertexId source, VertexId target, std::vector<VertexId>* path_out = nullptr,
                 std::size_t* nodes_settled_out = nullptr) const {
        if (nodes_settled_out != nullptr) {
            *nodes_settled_out = 0;
        }
        const std::size_t num_v = graph_.num_vertices();
        if (source >= num_v || target >= num_v) {
            if (path_out != nullptr) {
                path_out->clear();
            }
            return std::nullopt;
        }

        if (source == target) {
            if (path_out != nullptr) {
                *path_out = {source};
            }
            if (nodes_settled_out != nullptr) {
                *nodes_settled_out = 1;
            }
            return WeightType{0};
        }

        std::vector<WeightType> distances(num_v, INF_WEIGHT);
        std::vector<VertexId> predecessors(num_v, NO_PREDECESSOR);

        distances[source] = WeightType{0};

        d_ary_heap<4, WeightType, VertexId> heap(num_v);
        heap.push(source, WeightType{0});

        bool target_reached = false;
        std::size_t settled_count = 0;

        while (!heap.empty()) {
            auto [u, d_u] = heap.pop();
            ++settled_count;

            if (u == target) {
                target_reached = true;
                break;
            }

            if (d_u > distances[u]) {
                continue;
            }

            auto neighbors = graph_.out_neighbors(u);
            auto weights = graph_.out_weights(u);
            const std::size_t deg = neighbors.size();

            for (std::size_t i = 0; i < deg; ++i) {
                const VertexId v = neighbors[i];
                const WeightType w = weights[i];
                const WeightType new_dist = d_u + w;

                if (new_dist < distances[v]) {
                    distances[v] = new_dist;
                    predecessors[v] = u;
                    heap.push_or_decrease_key(v, new_dist);
                }
            }
        }

        if (nodes_settled_out != nullptr) {
            *nodes_settled_out = settled_count;
        }

        if (!target_reached) {
            if (path_out != nullptr) {
                path_out->clear();
            }
            return std::nullopt;
        }

        if (path_out != nullptr) {
            reconstruct_path(source, target, predecessors, *path_out);
        }

        return distances[target];
    }

    /**
     * @brief Bidirectional Dijkstra search with symmetric frontier termination.
     *
     * @param source Origin vertex index.
     * @param target Destination vertex index.
     * @param path_out Optional output vector to store reconstructed path vertices [source ...
     * target].
     * @return Shortest distance if reachable, or std::nullopt.
     */
    [[nodiscard]] std::optional<WeightType>
    dijkstra_bidirectional(VertexId source, VertexId target,
                           std::vector<VertexId>* path_out = nullptr) const {
        const std::size_t num_v = graph_.num_vertices();
        if (source >= num_v || target >= num_v) {
            if (path_out != nullptr) {
                path_out->clear();
            }
            return std::nullopt;
        }

        if (source == target) {
            if (path_out != nullptr) {
                *path_out = {source};
            }
            return WeightType{0};
        }

        std::vector<WeightType> dist_f(num_v, INF_WEIGHT);
        std::vector<WeightType> dist_r(num_v, INF_WEIGHT);
        std::vector<VertexId> pred_f(num_v, NO_PREDECESSOR);
        std::vector<VertexId> pred_r(num_v, NO_PREDECESSOR);

        dist_f[source] = WeightType{0};
        dist_r[target] = WeightType{0};

        d_ary_heap<4, WeightType, VertexId> q_f(num_v);
        d_ary_heap<4, WeightType, VertexId> q_r(num_v);

        q_f.push(source, WeightType{0});
        q_r.push(target, WeightType{0});

        WeightType mu = INF_WEIGHT;
        VertexId meeting_node = NO_PREDECESSOR;

        while (!q_f.empty() && !q_r.empty()) {
            // Stopping condition: if top_f + top_r >= mu, no strictly shorter path exists
            if (q_f.top_key() + q_r.top_key() >= mu) {
                break;
            }

            // Alternate directions balancing heap sizes
            if (q_f.size() <= q_r.size()) {
                auto [u, d_u] = q_f.pop();
                if (d_u <= dist_f[u]) {
                    auto neighbors = graph_.out_neighbors(u);
                    auto weights = graph_.out_weights(u);
                    const std::size_t deg = neighbors.size();

                    for (std::size_t i = 0; i < deg; ++i) {
                        const VertexId v = neighbors[i];
                        const WeightType w = weights[i];
                        const WeightType new_dist = d_u + w;

                        if (new_dist < dist_f[v]) {
                            dist_f[v] = new_dist;
                            pred_f[v] = u;
                            q_f.push_or_decrease_key(v, new_dist);

                            if (dist_r[v] != INF_WEIGHT && new_dist + dist_r[v] < mu) {
                                mu = new_dist + dist_r[v];
                                meeting_node = v;
                            }
                        }
                    }
                }
            } else {
                auto [u, d_u] = q_r.pop();
                if (d_u <= dist_r[u]) {
                    // For reverse search, explore incoming edges to u: (v -> u)
                    auto neighbors = graph_.in_neighbors(u);
                    auto weights = graph_.in_weights(u);
                    const std::size_t deg = neighbors.size();

                    for (std::size_t i = 0; i < deg; ++i) {
                        const VertexId v = neighbors[i];
                        const WeightType w = weights[i];
                        const WeightType new_dist = d_u + w;

                        if (new_dist < dist_r[v]) {
                            dist_r[v] = new_dist;
                            pred_r[v] = u; // In reverse tree, u is predecessor towards target
                            q_r.push_or_decrease_key(v, new_dist);

                            if (dist_f[v] != INF_WEIGHT && dist_f[v] + new_dist < mu) {
                                mu = dist_f[v] + new_dist;
                                meeting_node = v;
                            }
                        }
                    }
                }
            }
        }

        if (mu == INF_WEIGHT || meeting_node == NO_PREDECESSOR) {
            if (path_out != nullptr) {
                path_out->clear();
            }
            return std::nullopt;
        }

        if (path_out != nullptr) {
            reconstruct_bidirectional_path(source, target, meeting_node, pred_f, pred_r, *path_out);
        }

        return mu;
    }

    /**
     * @brief Precomputes distances for K landmark nodes using farthest-vertex selection.
     *
     * @param num_landmarks Number of landmark nodes to precompute.
     */
    void preprocess_landmarks(std::size_t num_landmarks = 16) {
        const std::size_t num_v = graph_.num_vertices();
        if (num_v == 0) {
            landmarks_.clear();
            landmark_from_.clear();
            landmark_to_.clear();
            return;
        }

        const std::size_t k = std::min(num_landmarks, num_v);
        if (k == 0) {
            landmarks_.clear();
            landmark_from_.clear();
            landmark_to_.clear();
            return;
        }

        landmarks_.clear();
        landmarks_.reserve(k);

        std::vector<bool> is_landmark(num_v, false);

        // Landmark selection: Farthest vertex placement (Goldberg & Harrelson 2005)
        std::vector<WeightType> min_dist_to_landmarks(num_v, INF_WEIGHT);

        // Initial landmark: node 0 (or farthest from node 0)
        auto initial_ssp = dijkstra(0);
        VertexId first_landmark = 0;
        WeightType max_d = -1.0;
        for (VertexId u = 0; u < num_v; ++u) {
            if (initial_ssp.distances[u] != INF_WEIGHT && initial_ssp.distances[u] > max_d) {
                max_d = initial_ssp.distances[u];
                first_landmark = u;
            }
        }
        landmarks_.push_back(first_landmark);
        is_landmark[first_landmark] = true;

        // Precompute forward and backward distances
        landmark_from_.resize(k);
        landmark_from_[0] = dijkstra(first_landmark).distances;
        for (VertexId u = 0; u < num_v; ++u) {
            min_dist_to_landmarks[u] = landmark_from_[0][u];
        }

        if (graph_.is_directed()) {
            landmark_to_.resize(k);
            landmark_to_[0] = compute_reverse_dijkstra(first_landmark);
        }

        for (std::size_t i = 1; i < k; ++i) {
            VertexId best_candidate = NO_PREDECESSOR;
            WeightType best_dist = -1.0;

            for (VertexId u = 0; u < num_v; ++u) {
                if (!is_landmark[u] && min_dist_to_landmarks[u] != INF_WEIGHT &&
                    min_dist_to_landmarks[u] > best_dist) {
                    best_dist = min_dist_to_landmarks[u];
                    best_candidate = u;
                }
            }

            // If all reachable vertices from current landmarks are selected, pick any unchosen
            // vertex (e.g. from an unexplored disconnected component)
            if (best_candidate == NO_PREDECESSOR) {
                for (VertexId u = 0; u < num_v; ++u) {
                    if (!is_landmark[u]) {
                        best_candidate = u;
                        break;
                    }
                }
            }

            if (best_candidate == NO_PREDECESSOR) {
                break;
            }

            is_landmark[best_candidate] = true;
            landmarks_.push_back(best_candidate);
            landmark_from_[i] = dijkstra(best_candidate).distances;

            for (VertexId u = 0; u < num_v; ++u) {
                if (landmark_from_[i][u] < min_dist_to_landmarks[u]) {
                    min_dist_to_landmarks[u] = landmark_from_[i][u];
                }
            }

            if (graph_.is_directed()) {
                landmark_to_[i] = compute_reverse_dijkstra(best_candidate);
            }
        }
    }

    /**
     * @brief Point-to-point ALT search (A* with Landmark Triangle inequality heuristic).
     *
     * @param source Origin vertex index.
     * @param target Destination vertex index.
     * @param path_out Optional output vector to store reconstructed path vertices [source ...
     * target].
     * @return Shortest distance if reachable, or std::nullopt.
     */
    [[nodiscard]] std::optional<WeightType>
    alt_search(VertexId source, VertexId target, std::vector<VertexId>* path_out = nullptr,
               std::size_t* nodes_settled_out = nullptr) const {
        if (nodes_settled_out != nullptr) {
            *nodes_settled_out = 0;
        }

        if (!has_landmarks()) {
            return dijkstra_p2p(source, target, path_out, nodes_settled_out);
        }

        const std::size_t num_v = graph_.num_vertices();
        if (source >= num_v || target >= num_v) {
            if (path_out != nullptr) {
                path_out->clear();
            }
            return std::nullopt;
        }

        if (source == target) {
            if (path_out != nullptr) {
                *path_out = {source};
            }
            if (nodes_settled_out != nullptr) {
                *nodes_settled_out = 1;
            }
            return WeightType{0};
        }

        std::vector<WeightType> g_score(num_v, INF_WEIGHT);
        std::vector<VertexId> predecessors(num_v, NO_PREDECESSOR);

        g_score[source] = WeightType{0};

        // Heap stores (f_score = g_score + heuristic)
        d_ary_heap<4, WeightType, VertexId> heap(num_v);
        const WeightType h_source = landmark_heuristic(source, target);
        heap.push(source, h_source);

        bool target_reached = false;
        std::size_t settled_count = 0;

        while (!heap.empty()) {
            auto [u, f_u] = heap.pop();
            ++settled_count;

            if (u == target) {
                target_reached = true;
                break;
            }

            const WeightType g_u = g_score[u];
            if (g_u == INF_WEIGHT) {
                continue;
            }

            auto neighbors = graph_.out_neighbors(u);
            auto weights = graph_.out_weights(u);
            const std::size_t deg = neighbors.size();

            for (std::size_t i = 0; i < deg; ++i) {
                const VertexId v = neighbors[i];
                const WeightType w = weights[i];
                const WeightType tentative_g = g_u + w;

                if (tentative_g < g_score[v]) {
                    g_score[v] = tentative_g;
                    predecessors[v] = u;

                    const WeightType h_v = landmark_heuristic(v, target);
                    const WeightType f_v = tentative_g + h_v;
                    heap.push_or_decrease_key(v, f_v);
                }
            }
        }

        if (nodes_settled_out != nullptr) {
            *nodes_settled_out = settled_count;
        }

        if (!target_reached) {
            if (path_out != nullptr) {
                path_out->clear();
            }
            return std::nullopt;
        }

        if (path_out != nullptr) {
            reconstruct_path(source, target, predecessors, *path_out);
        }

        return g_score[target];
    }

    /**
     * @brief Bidirectional ALT search with symmetric Pohl stopping condition.
     *
     * @param source Origin vertex index.
     * @param target Destination vertex index.
     * @param path_out Optional output vector to store reconstructed path vertices.
     * @return Shortest distance if reachable, or std::nullopt.
     */
    [[nodiscard]] std::optional<WeightType>
    alt_search_bidirectional(VertexId source, VertexId target,
                             std::vector<VertexId>* path_out = nullptr) const {
        if (!has_landmarks()) {
            return dijkstra_bidirectional(source, target, path_out);
        }

        const std::size_t num_v = graph_.num_vertices();
        if (source >= num_v || target >= num_v) {
            if (path_out != nullptr) {
                path_out->clear();
            }
            return std::nullopt;
        }

        if (source == target) {
            if (path_out != nullptr) {
                *path_out = {source};
            }
            return WeightType{0};
        }

        std::vector<WeightType> g_f(num_v, INF_WEIGHT);
        std::vector<WeightType> g_r(num_v, INF_WEIGHT);
        std::vector<VertexId> pred_f(num_v, NO_PREDECESSOR);
        std::vector<VertexId> pred_r(num_v, NO_PREDECESSOR);

        g_f[source] = WeightType{0};
        g_r[target] = WeightType{0};

        d_ary_heap<4, WeightType, VertexId> q_f(num_v);
        d_ary_heap<4, WeightType, VertexId> q_r(num_v);

        q_f.push(source, landmark_heuristic(source, target));
        q_r.push(target, landmark_heuristic_reverse(target, source));

        WeightType mu = INF_WEIGHT;
        VertexId meeting_node = NO_PREDECESSOR;

        while (!q_f.empty() && !q_r.empty()) {
            // Stopping condition: when both forward and reverse minimum keys exceed mu,
            // no unexplored path can be shorter than mu.
            if (q_f.top_key() >= mu && q_r.top_key() >= mu) {
                break;
            }

            // Alternation policy: expand the queue with smaller minimum key
            if (q_f.top_key() <= q_r.top_key()) {
                auto [u, f_u] = q_f.pop();
                const WeightType g_u = g_f[u];
                if (g_u != INF_WEIGHT) {
                    auto neighbors = graph_.out_neighbors(u);
                    auto weights = graph_.out_weights(u);
                    const std::size_t deg = neighbors.size();

                    for (std::size_t i = 0; i < deg; ++i) {
                        const VertexId v = neighbors[i];
                        const WeightType w = weights[i];
                        const WeightType tentative_g = g_u + w;

                        if (tentative_g < g_f[v]) {
                            g_f[v] = tentative_g;
                            pred_f[v] = u;

                            const WeightType f_v = tentative_g + landmark_heuristic(v, target);
                            q_f.push_or_decrease_key(v, f_v);

                            if (g_r[v] != INF_WEIGHT && tentative_g + g_r[v] < mu) {
                                mu = tentative_g + g_r[v];
                                meeting_node = v;
                            }
                        }
                    }
                }
            } else {
                auto [u, f_u] = q_r.pop();
                const WeightType g_u = g_r[u];
                if (g_u != INF_WEIGHT) {
                    auto neighbors = graph_.in_neighbors(u);
                    auto weights = graph_.in_weights(u);
                    const std::size_t deg = neighbors.size();

                    for (std::size_t i = 0; i < deg; ++i) {
                        const VertexId v = neighbors[i];
                        const WeightType w = weights[i];
                        const WeightType tentative_g = g_u + w;

                        if (tentative_g < g_r[v]) {
                            g_r[v] = tentative_g;
                            pred_r[v] = u;

                            const WeightType f_v =
                                tentative_g + landmark_heuristic_reverse(v, source);
                            q_r.push_or_decrease_key(v, f_v);

                            if (g_f[v] != INF_WEIGHT && g_f[v] + tentative_g < mu) {
                                mu = g_f[v] + tentative_g;
                                meeting_node = v;
                            }
                        }
                    }
                }
            }
        }

        if (mu == INF_WEIGHT || meeting_node == NO_PREDECESSOR) {
            if (path_out != nullptr) {
                path_out->clear();
            }
            return std::nullopt;
        }

        if (path_out != nullptr) {
            reconstruct_bidirectional_path(source, target, meeting_node, pred_f, pred_r, *path_out);
        }

        return mu;
    }

    [[nodiscard]] const std::vector<VertexId>& landmarks() const noexcept {
        return landmarks_;
    }

    [[nodiscard]] bool has_landmarks() const noexcept {
        return !landmarks_.empty();
    }

private:
    std::vector<WeightType> compute_reverse_dijkstra(VertexId target) const {
        const std::size_t num_v = graph_.num_vertices();
        std::vector<WeightType> distances(num_v, INF_WEIGHT);
        if (target >= num_v) {
            return distances;
        }

        distances[target] = WeightType{0};

        d_ary_heap<4, WeightType, VertexId> heap(num_v);
        heap.push(target, WeightType{0});

        while (!heap.empty()) {
            auto [u, d_u] = heap.pop();
            if (d_u > distances[u]) {
                continue;
            }

            auto in_nbrs = graph_.in_neighbors(u);
            auto in_wts = graph_.in_weights(u);
            const std::size_t deg = in_nbrs.size();

            for (std::size_t i = 0; i < deg; ++i) {
                const VertexId v = in_nbrs[i];
                const WeightType w = in_wts[i];
                const WeightType new_dist = d_u + w;

                if (new_dist < distances[v]) {
                    distances[v] = new_dist;
                    heap.push_or_decrease_key(v, new_dist);
                }
            }
        }

        return distances;
    }

    [[nodiscard]] WeightType landmark_heuristic(VertexId u, VertexId t) const noexcept {
        WeightType max_h = WeightType{0};
        const std::size_t k = landmarks_.size();

        for (std::size_t i = 0; i < k; ++i) {
            // From landmark L_i: d(L, t) - d(L, u)
            const WeightType d_L_t = landmark_from_[i][t];
            const WeightType d_L_u = landmark_from_[i][u];
            if (d_L_t != INF_WEIGHT && d_L_u != INF_WEIGHT && d_L_t > d_L_u) {
                const WeightType h1 = d_L_t - d_L_u;
                if (h1 > max_h) {
                    max_h = h1;
                }
            }

            // To landmark L_i: d(u, L) - d(t, L)
            const auto& dist_to = graph_.is_directed() ? landmark_to_[i] : landmark_from_[i];
            const WeightType d_u_L = dist_to[u];
            const WeightType d_t_L = dist_to[t];
            if (d_u_L != INF_WEIGHT && d_t_L != INF_WEIGHT && d_u_L > d_t_L) {
                const WeightType h2 = d_u_L - d_t_L;
                if (h2 > max_h) {
                    max_h = h2;
                }
            }
        }

        return max_h;
    }

    [[nodiscard]] WeightType landmark_heuristic_reverse(VertexId u, VertexId s) const noexcept {
        // Reverse heuristic estimates distance from s to u in reverse search
        // (which is distance from u to s in forward graph)
        return landmark_heuristic(s, u);
    }

    static void reconstruct_path(VertexId source, VertexId target,
                                 const std::vector<VertexId>& predecessors,
                                 std::vector<VertexId>& path_out) {
        path_out.clear();
        VertexId curr = target;
        while (curr != NO_PREDECESSOR) {
            path_out.push_back(curr);
            if (curr == source) {
                break;
            }
            curr = predecessors[curr];
        }
        std::reverse(path_out.begin(), path_out.end());
    }

    static void reconstruct_bidirectional_path(VertexId source, VertexId target,
                                               VertexId meeting_node,
                                               const std::vector<VertexId>& pred_f,
                                               const std::vector<VertexId>& pred_r,
                                               std::vector<VertexId>& path_out) {
        path_out.clear();

        // 1. Forward half: meeting_node back to source
        std::vector<VertexId> f_path;
        VertexId curr = meeting_node;
        while (curr != NO_PREDECESSOR) {
            f_path.push_back(curr);
            if (curr == source) {
                break;
            }
            curr = pred_f[curr];
        }
        std::reverse(f_path.begin(), f_path.end());

        // 2. Backward half: from meeting_node towards target
        std::vector<VertexId> r_path;
        curr = pred_r[meeting_node];
        while (curr != NO_PREDECESSOR) {
            r_path.push_back(curr);
            if (curr == target) {
                break;
            }
            curr = pred_r[curr];
        }

        path_out = std::move(f_path);
        path_out.insert(path_out.end(), r_path.begin(), r_path.end());
    }

    const CSRGraph<WeightType>& graph_;
    std::vector<VertexId> landmarks_;
    std::vector<std::vector<WeightType>> landmark_from_;
    std::vector<std::vector<WeightType>> landmark_to_;
};

} // namespace algoat::graph

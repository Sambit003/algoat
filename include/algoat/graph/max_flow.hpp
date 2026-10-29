/**
 * @file max_flow.hpp
 * @brief Production-grade Maximum Network Flow & Minimum Cut algorithms.
 *
 * Implements:
 * 1. Dinic's Algorithm with BFS level graphs and current-arc DFS blocking flows (O(V^2 E) general,
 *    O(E sqrt(V)) unit networks).
 * 2. Highest-Label Push-Relabel (HLPR) with Cherkassky-Goldberg Global Relabeling (periodic
 * backward BFS) and Gap Relabeling heuristics (O(V^2 sqrt(E))), maintaining excess flow in buckets
 *    indexed by height h in [0, 2V-1].
 */

#pragma once

#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace algoat::graph {

/**
 * @struct FlowEdge
 * @brief Representation of a directed network flow edge with capacity and computed flow.
 */
template <typename FlowType = int64_t> struct FlowEdge {
    VertexId from{0};
    VertexId to{0};
    FlowType capacity{0};
    FlowType flow{0};
};

/**
 * @struct MaxFlowResult
 * @brief Output container for maximum network flow and minimum s-t cut.
 */
template <typename FlowType = int64_t> struct MaxFlowResult {
    FlowType max_flow{0};
    std::vector<FlowEdge<FlowType>> residual_edges;
    std::vector<VertexId> min_cut_source_set; // Vertices on S-side of min-cut
};

/**
 * @class MaxFlowEngine
 * @brief High-performance network flow engine for directed capacity networks.
 */
template <typename FlowType = int64_t> class MaxFlowEngine {
    static_assert(std::integral<FlowType>, "FlowType must be an integral type");

public:
    static constexpr EdgeIndex kNullEdge = std::numeric_limits<EdgeIndex>::max();
    static constexpr VertexId kNullVertex = std::numeric_limits<VertexId>::max();

    explicit MaxFlowEngine(std::size_t num_vertices)
        : num_vertices_(num_vertices), head_(num_vertices, kNullEdge) {}

    /**
     * @brief Adds a directed capacity edge (u -> v). Automatically allocates the backward
     * residual edge (v -> u) paired via bitwise XOR (e ^ 1).
     *
     * @param from Origin vertex ID.
     * @param to Destination vertex ID.
     * @param capacity Maximum allowable flow along this edge.
     */
    void add_edge(VertexId from, VertexId to, FlowType capacity) {
        if (from >= num_vertices_ || to >= num_vertices_) {
            throw std::out_of_range("Vertex index out of bounds in MaxFlowEngine::add_edge");
        }

        const auto forward_idx = static_cast<EdgeIndex>(to_.size());

        // Forward edge (even index)
        to_.push_back(to);
        cap_.push_back(capacity);
        flow_.push_back(0);
        next_.push_back(head_[from]);
        head_[from] = forward_idx;

        // Backward residual edge (odd index: forward_idx ^ 1)
        to_.push_back(from);
        cap_.push_back(0);
        flow_.push_back(0);
        next_.push_back(head_[to]);
        head_[to] = forward_idx + 1;
    }

    [[nodiscard]] std::size_t num_vertices() const noexcept {
        return num_vertices_;
    }
    [[nodiscard]] std::size_t num_edges() const noexcept {
        return to_.size() / 2;
    }

    /**
     * @brief Dinic's Algorithm for Maximum Flow.
     *
     * Constructs layered level graphs via BFS and computes blocking flows via DFS with
     * current-arc optimization.
     *
     * @param source Flow network source vertex.
     * @param sink Flow network sink vertex.
     * @return MaxFlowResult containing max flow value, residual flow edges, and min-cut source set.
     */
    [[nodiscard]] MaxFlowResult<FlowType> dinic(VertexId source, VertexId sink) {
        if (num_vertices_ == 0 || source >= num_vertices_ || sink >= num_vertices_ ||
            source == sink) {
            return trivial_result(source);
        }

        std::fill(flow_.begin(), flow_.end(), 0);

        std::vector<int64_t> level(num_vertices_, -1);
        std::vector<EdgeIndex> work_head(num_vertices_, kNullEdge);
        std::vector<VertexId> bfs_queue(num_vertices_);

        FlowType total_flow = 0;
        constexpr FlowType kFlowInf = std::numeric_limits<FlowType>::max() / 2;

        auto bfs_level_graph = [&]() -> bool {
            std::fill(level.begin(), level.end(), -1);
            level[source] = 0;
            std::size_t queue_head = 0;
            std::size_t queue_tail = 0;
            bfs_queue[queue_tail++] = source;

            while (queue_head < queue_tail) {
                const VertexId u = bfs_queue[queue_head++];
                for (EdgeIndex e = head_[u]; e != kNullEdge; e = next_[e]) {
                    const VertexId v = to_[e];
                    if (cap_[e] > flow_[e] && level[v] == -1) {
                        level[v] = level[u] + 1;
                        bfs_queue[queue_tail++] = v;
                    }
                }
            }
            return level[sink] != -1;
        };

        auto dfs_blocking_flow = [&](auto& self, VertexId u, FlowType pushed) -> FlowType {
            if (pushed == 0 || u == sink) {
                return pushed;
            }

            for (EdgeIndex& e = work_head[u]; e != kNullEdge; e = next_[e]) {
                const VertexId v = to_[e];
                const FlowType res_cap = cap_[e] - flow_[e];

                if (res_cap > 0 && level[v] == level[u] + 1) {
                    const FlowType transferred = self(self, v, std::min(pushed, res_cap));
                    if (transferred > 0) {
                        flow_[e] += transferred;
                        flow_[e ^ 1] -= transferred;
                        return transferred;
                    }
                }
            }
            return 0;
        };

        while (bfs_level_graph()) {
            work_head = head_;
            while (true) {
                const FlowType pushed = dfs_blocking_flow(dfs_blocking_flow, source, kFlowInf);
                if (pushed == 0) {
                    break;
                }
                total_flow += pushed;
            }
        }

        MaxFlowResult<FlowType> result;
        result.max_flow = total_flow;
        result.min_cut_source_set = extract_min_cut_source_set(source);
        result.residual_edges = extract_flow_edges();
        return result;
    }

    /**
     * @brief Highest-Label Push-Relabel (HLPR) algorithm with Cherkassky-Goldberg Global
     * Relabeling and Gap Relabeling heuristics.
     *
     * Maintains excess flow in height buckets indexed by h in [0, 2V-1], achieving O(V^2 sqrt(E))
     * runtime with complete exact flow conservation.
     *
     * @param source Flow network source vertex.
     * @param sink Flow network sink vertex.
     * @return MaxFlowResult containing max flow value, residual flow edges, and min-cut source set.
     */
    [[nodiscard]] MaxFlowResult<FlowType> push_relabel(VertexId source, VertexId sink) {
        if (num_vertices_ == 0 || source >= num_vertices_ || sink >= num_vertices_ ||
            source == sink) {
            return trivial_result(source);
        }

        std::fill(flow_.begin(), flow_.end(), 0);

        const auto n = static_cast<VertexId>(num_vertices_);
        const std::size_t max_bucket_count = 2 * num_vertices_ + 2;

        std::vector<FlowType> excess(num_vertices_, 0);
        std::vector<VertexId> height(num_vertices_, 0);
        std::vector<std::size_t> count_height(max_bucket_count, 0);
        std::vector<EdgeIndex> current_arc = head_;

        // Intrusive height bucket priority queue for O(1) active vertex management
        struct HeightBucketQueue {
            std::vector<VertexId> bucket_head;
            std::vector<VertexId> node_prev;
            std::vector<VertexId> node_next;
            std::vector<uint8_t> in_bucket;
            VertexId max_height{0};

            HeightBucketQueue(std::size_t num_v, std::size_t max_h)
                : bucket_head(max_h, kNullVertex), node_prev(num_v, kNullVertex),
                  node_next(num_v, kNullVertex), in_bucket(num_v, 0) {}

            void clear() {
                std::fill(bucket_head.begin(), bucket_head.end(), kNullVertex);
                std::fill(in_bucket.begin(), in_bucket.end(), 0);
                max_height = 0;
            }

            void insert(VertexId u, VertexId h) {
                if (in_bucket[u] || h >= bucket_head.size()) {
                    return;
                }
                node_prev[u] = kNullVertex;
                node_next[u] = bucket_head[h];
                if (bucket_head[h] != kNullVertex) {
                    node_prev[bucket_head[h]] = u;
                }
                bucket_head[h] = u;
                in_bucket[u] = 1;
                if (h > max_height) {
                    max_height = h;
                }
            }

            void remove(VertexId u, VertexId h) {
                if (!in_bucket[u] || h >= bucket_head.size()) {
                    return;
                }
                if (node_prev[u] != kNullVertex) {
                    node_next[node_prev[u]] = node_next[u];
                } else {
                    bucket_head[h] = node_next[u];
                }
                if (node_next[u] != kNullVertex) {
                    node_prev[node_next[u]] = node_prev[u];
                }
                in_bucket[u] = 0;
                node_prev[u] = kNullVertex;
                node_next[u] = kNullVertex;
            }

            [[nodiscard]] VertexId pop_highest() {
                while (max_height > 0 && bucket_head[max_height] == kNullVertex) {
                    max_height--;
                }
                if (bucket_head[max_height] == kNullVertex) {
                    return kNullVertex;
                }
                const VertexId u = bucket_head[max_height];
                remove(u, max_height);
                return u;
            }

            [[nodiscard]] bool empty() {
                while (max_height > 0 && bucket_head[max_height] == kNullVertex) {
                    max_height--;
                }
                return bucket_head[max_height] == kNullVertex;
            }
        } buckets(num_vertices_, max_bucket_count);

        std::size_t push_counter = 0;
        const std::size_t global_relabel_freq =
            std::max<std::size_t>(num_vertices_, to_.size() / 2);

        // Global Relabeling: backward BFS from sink t resets exact residual distances
        auto global_relabel = [&]() {
            push_counter = 0;
            std::vector<VertexId> new_height(num_vertices_, n);
            std::fill(count_height.begin(), count_height.end(), 0);
            buckets.clear();

            std::vector<VertexId> bfs_q(num_vertices_);
            std::size_t queue_head = 0;
            std::size_t queue_tail = 0;

            std::vector<uint8_t> visited(num_vertices_, 0);
            visited[sink] = 1;
            new_height[sink] = 0;
            bfs_q[queue_tail++] = sink;

            while (queue_head < queue_tail) {
                const VertexId v = bfs_q[queue_head++];

                for (EdgeIndex e = head_[v]; e != kNullEdge; e = next_[e]) {
                    const VertexId u = to_[e];
                    // Edge u -> v has capacity in residual graph if cap_[e ^ 1] > flow_[e ^ 1]
                    if (cap_[e ^ 1] > flow_[e ^ 1] && !visited[u]) {
                        visited[u] = 1;
                        new_height[u] = new_height[v] + 1;
                        bfs_q[queue_tail++] = u;
                    }
                }
            }

            for (VertexId u = 0; u < n; ++u) {
                if (u == sink) {
                    height[u] = 0;
                } else if (u == source) {
                    height[u] = n;
                } else if (visited[u]) {
                    height[u] = new_height[u];
                } else {
                    height[u] = std::max(height[u], n);
                }
                if (height[u] < count_height.size()) {
                    count_height[height[u]]++;
                }
            }

            current_arc = head_;

            for (VertexId u = 0; u < n; ++u) {
                if (u != source && u != sink && excess[u] > 0 && height[u] < 2 * n) {
                    buckets.insert(u, height[u]);
                }
            }
        };

        auto push = [&](VertexId u, EdgeIndex e) {
            const VertexId v = to_[e];
            const FlowType push_amt = std::min(excess[u], cap_[e] - flow_[e]);
            if (push_amt <= 0 || height[u] != height[v] + 1) {
                return;
            }

            push_counter++;
            flow_[e] += push_amt;
            flow_[e ^ 1] -= push_amt;
            excess[u] -= push_amt;
            excess[v] += push_amt;

            if (excess[v] > 0 && v != source && v != sink && !buckets.in_bucket[v] &&
                height[v] < 2 * n) {
                buckets.insert(v, height[v]);
            }
        };

        auto relabel = [&](VertexId u) {
            VertexId min_h = 2 * n;
            for (EdgeIndex e = head_[u]; e != kNullEdge; e = next_[e]) {
                if (cap_[e] > flow_[e]) {
                    min_h = std::min(min_h, height[to_[e]]);
                }
            }

            const VertexId old_h = height[u];
            count_height[old_h]--;
            height[u] = min_h + 1;
            if (height[u] < count_height.size()) {
                count_height[height[u]]++;
            }
            current_arc[u] = head_[u];

            // Gap Relabeling Heuristic: If count at old_h drops to 0 (for old_h < n),
            // all nodes with old_h < h < n cannot reach sink t; lift them to n + 1
            if (old_h < n && count_height[old_h] == 0) {
                for (VertexId v = 0; v < n; ++v) {
                    if (v != source && v != sink && height[v] > old_h && height[v] < n) {
                        count_height[height[v]]--;
                        buckets.remove(v, height[v]);
                        height[v] = n + 1;
                        count_height[height[v]]++;
                        if (excess[v] > 0) {
                            buckets.insert(v, height[v]);
                        }
                    }
                }
            }
        };

        auto discharge = [&](VertexId u) {
            while (excess[u] > 0) {
                if (current_arc[u] == kNullEdge) {
                    relabel(u);
                    if (height[u] >= 2 * n) {
                        break;
                    }
                } else {
                    const EdgeIndex e = current_arc[u];
                    const VertexId v = to_[e];
                    if (cap_[e] > flow_[e] && height[u] == height[v] + 1) {
                        push(u, e);
                    } else {
                        current_arc[u] = next_[e];
                    }
                }
            }
        };

        // Initialize distance labels
        global_relabel();

        // Push maximum capacity flow out of source
        for (EdgeIndex e = head_[source]; e != kNullEdge; e = next_[e]) {
            const FlowType c = cap_[e];
            if (c > 0) {
                flow_[e] += c;
                flow_[e ^ 1] -= c;
                excess[to_[e]] += c;
                excess[source] -= c;
                push_counter++;
            }
        }

        // Populate active buckets
        for (VertexId u = 0; u < n; ++u) {
            if (u != source && u != sink && excess[u] > 0 && height[u] < 2 * n) {
                buckets.insert(u, height[u]);
            }
        }

        // Highest-Label Push-Relabel Loop
        while (!buckets.empty()) {
            const VertexId u = buckets.pop_highest();
            if (u == kNullVertex) {
                break;
            }
            discharge(u);

            if (excess[u] > 0 && height[u] < 2 * n) {
                buckets.insert(u, height[u]);
            }

            if (push_counter >= global_relabel_freq) {
                global_relabel();
            }
        }

        MaxFlowResult<FlowType> result;
        result.max_flow = excess[sink];
        result.min_cut_source_set = extract_min_cut_source_set(source);
        result.residual_edges = extract_flow_edges();
        return result;
    }

private:
    std::size_t num_vertices_{0};
    std::vector<EdgeIndex> head_;
    std::vector<VertexId> to_;
    std::vector<EdgeIndex> next_;
    std::vector<FlowType> cap_;
    std::vector<FlowType> flow_;

    [[nodiscard]] MaxFlowResult<FlowType> trivial_result(VertexId source) const {
        MaxFlowResult<FlowType> res;
        res.max_flow = 0;
        if (num_vertices_ > 0 && source < num_vertices_) {
            res.min_cut_source_set.push_back(source);
        }
        res.residual_edges = extract_flow_edges();
        return res;
    }

    [[nodiscard]] std::vector<VertexId> extract_min_cut_source_set(VertexId source) const {
        std::vector<VertexId> s_set;
        if (num_vertices_ == 0 || source >= num_vertices_) {
            return s_set;
        }

        std::vector<uint8_t> visited(num_vertices_, 0);
        std::vector<VertexId> bfs_q(num_vertices_);
        std::size_t queue_head = 0;
        std::size_t queue_tail = 0;

        visited[source] = 1;
        bfs_q[queue_tail++] = source;

        while (queue_head < queue_tail) {
            const VertexId u = bfs_q[queue_head++];
            s_set.push_back(u);

            for (EdgeIndex e = head_[u]; e != kNullEdge; e = next_[e]) {
                const VertexId v = to_[e];
                if (cap_[e] > flow_[e] && !visited[v]) {
                    visited[v] = 1;
                    bfs_q[queue_tail++] = v;
                }
            }
        }
        std::sort(s_set.begin(), s_set.end());
        return s_set;
    }

    [[nodiscard]] std::vector<FlowEdge<FlowType>> extract_flow_edges() const {
        std::vector<FlowEdge<FlowType>> edges;
        edges.reserve(to_.size() / 2);
        for (std::size_t e = 0; e < to_.size(); e += 2) {
            const VertexId u = to_[e ^ 1];
            const VertexId v = to_[e];
            edges.push_back(FlowEdge<FlowType>{
                .from = u,
                .to = v,
                .capacity = cap_[e],
                .flow = flow_[e],
            });
        }
        return edges;
    }
};

} // namespace algoat::graph

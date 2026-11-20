#pragma once

#include "algoat/graph/link_cut_tree.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace algoat::graph::testing {

template <typename ValueType = int64_t> class ForestOracle {
public:
    explicit ForestOracle(std::size_t n)
        : n_(n), parent_(n, kNull), adj_(n), values_(n, ValueType{}), visit_stamp_(n, 0),
          bfs_parent_(n, kNull), queue_(n, 0) {}

    [[nodiscard]] bool connected(std::size_t u, std::size_t v) {
        if (u == v) {
            return true;
        }
        return run_bfs(u, v);
    }

    [[nodiscard]] std::size_t find_root(std::size_t u) const {
        while (parent_[u] != kNull) {
            u = parent_[u];
        }
        return u;
    }

    [[nodiscard]] bool is_root(std::size_t u) const noexcept {
        return parent_[u] == kNull;
    }

    [[nodiscard]] bool has_edge(std::size_t u, std::size_t v) const noexcept {
        return parent_[u] == v || parent_[v] == u;
    }

    [[nodiscard]] std::vector<std::size_t> all_roots() const {
        std::vector<std::size_t> roots(n_);
        for (std::size_t u = 0; u < n_; ++u) {
            roots[u] = find_root(u);
        }
        return roots;
    }

    void link(std::size_t u, std::size_t v) {
        parent_[u] = v;
        adj_[u].push_back(v);
        adj_[v].push_back(u);
    }

    void cut(std::size_t u, std::size_t v) {
        if (parent_[u] == v) {
            parent_[u] = kNull;
        } else {
            parent_[v] = kNull;
        }
        erase_neighbor(adj_[u], v);
        erase_neighbor(adj_[v], u);
    }

    void make_root(std::size_t u) {
        const std::size_t stamp = ++stamp_counter_;
        std::size_t head = 0;
        std::size_t tail = 0;
        visit_stamp_[u] = stamp;
        parent_[u] = kNull;
        queue_[tail++] = u;

        while (head < tail) {
            const std::size_t cur = queue_[head++];
            for (const std::size_t nxt : adj_[cur]) {
                if (visit_stamp_[nxt] != stamp) {
                    visit_stamp_[nxt] = stamp;
                    parent_[nxt] = cur;
                    queue_[tail++] = nxt;
                }
            }
        }
    }

    void update_vertex(std::size_t u, ValueType val) {
        values_[u] = std::move(val);
    }

    [[nodiscard]] std::pair<ValueType, ValueType> query_path(std::size_t u, std::size_t v) {
        if (u == v) {
            return {values_[u], values_[u]};
        }
        if (!run_bfs(u, v)) {
            throw std::logic_error("ForestOracle::query_path called on disconnected vertices");
        }
        ValueType sum = values_[v];
        ValueType mn = values_[v];
        for (std::size_t cur = bfs_parent_[v]; cur != kNull; cur = bfs_parent_[cur]) {
            sum = sum + values_[cur];
            if (values_[cur] < mn) {
                mn = values_[cur];
            }
            if (cur == u) {
                break;
            }
        }
        return {sum, mn};
    }

private:
    static constexpr std::size_t kNull = std::numeric_limits<std::size_t>::max();

    static void erase_neighbor(std::vector<std::size_t>& list, std::size_t target) {
        for (std::size_t i = 0; i < list.size(); ++i) {
            if (list[i] == target) {
                list[i] = list.back();
                list.pop_back();
                return;
            }
        }
    }

    bool run_bfs(std::size_t src, std::size_t dst) {
        const std::size_t stamp = ++stamp_counter_;
        std::size_t head = 0;
        std::size_t tail = 0;
        visit_stamp_[src] = stamp;
        bfs_parent_[src] = kNull;
        queue_[tail++] = src;

        while (head < tail) {
            const std::size_t cur = queue_[head++];
            if (cur == dst) {
                return true;
            }
            for (const std::size_t nxt : adj_[cur]) {
                if (visit_stamp_[nxt] != stamp) {
                    visit_stamp_[nxt] = stamp;
                    bfs_parent_[nxt] = cur;
                    queue_[tail++] = nxt;
                }
            }
        }
        return false;
    }

    std::size_t n_{0};
    std::vector<std::size_t> parent_;
    std::vector<std::vector<std::size_t>> adj_;
    std::vector<ValueType> values_;
    std::vector<std::size_t> visit_stamp_;
    std::vector<std::size_t> bfs_parent_;
    std::vector<std::size_t> queue_;
    std::size_t stamp_counter_{0};
};

struct DynamicMstEdgeValue {
    int64_t weight{0};
    std::size_t edge_slot{std::numeric_limits<std::size_t>::max()};
    bool is_edge{false};

    [[nodiscard]] DynamicMstEdgeValue operator+(const DynamicMstEdgeValue& rhs) const noexcept {
        return DynamicMstEdgeValue{weight + rhs.weight, std::numeric_limits<std::size_t>::max(),
                                   false};
    }

    // Reversed ordering so query_path_min yields the heaviest active tree edge
    [[nodiscard]] bool operator<(const DynamicMstEdgeValue& rhs) const noexcept {
        if (is_edge != rhs.is_edge) {
            return is_edge;
        }
        if (!is_edge) {
            return false;
        }
        if (weight != rhs.weight) {
            return weight > rhs.weight;
        }
        return edge_slot < rhs.edge_slot;
    }
};

class DynamicMstMaintainer {
public:
    explicit DynamicMstMaintainer(std::size_t num_vertices)
        : n_(num_vertices), lct_(num_vertices > 1 ? 2 * num_vertices - 1 : num_vertices),
          components_(num_vertices), edge_u_(num_vertices > 1 ? num_vertices - 1 : 0, 0),
          edge_v_(num_vertices > 1 ? num_vertices - 1 : 0, 0),
          free_slots_(num_vertices > 1 ? num_vertices - 1 : 0) {
        std::iota(free_slots_.begin(), free_slots_.end(), std::size_t{0});
    }

    bool insert_edge(std::size_t u, std::size_t v, int64_t weight) {
        if (u >= n_ || v >= n_) {
            throw std::out_of_range("DynamicMstMaintainer: vertex out of range");
        }
        if (u == v) {
            return false;
        }
        if (!lct_.connected(u, v)) {
            const std::size_t slot = free_slots_.back();
            free_slots_.pop_back();
            const std::size_t edge_node = n_ + slot;
            edge_u_[slot] = u;
            edge_v_[slot] = v;
            lct_.update_vertex(edge_node, DynamicMstEdgeValue{weight, slot, true});
            lct_.make_root(u);
            lct_.link(u, edge_node);
            lct_.link(edge_node, v);
            total_weight_ += weight;
            --components_;
            return true;
        }

        const DynamicMstEdgeValue heaviest = lct_.query_path_min(u, v);
        if (heaviest.is_edge && weight < heaviest.weight) {
            const std::size_t slot = heaviest.edge_slot;
            const std::size_t edge_node = n_ + slot;
            lct_.cut(edge_u_[slot], edge_node);
            lct_.cut(edge_v_[slot], edge_node);

            edge_u_[slot] = u;
            edge_v_[slot] = v;
            lct_.update_vertex(edge_node, DynamicMstEdgeValue{weight, slot, true});
            lct_.make_root(u);
            lct_.link(u, edge_node);
            lct_.link(edge_node, v);
            total_weight_ += weight - heaviest.weight;
            return true;
        }
        return false;
    }

    [[nodiscard]] int64_t total_weight() const noexcept {
        return total_weight_;
    }

    [[nodiscard]] std::size_t components_count() const noexcept {
        return components_;
    }

    [[nodiscard]] bool connected(std::size_t u, std::size_t v) {
        return lct_.connected(u, v);
    }

private:
    std::size_t n_{0};
    LinkCutTree<DynamicMstEdgeValue> lct_;
    int64_t total_weight_{0};
    std::size_t components_{0};
    std::vector<std::size_t> edge_u_;
    std::vector<std::size_t> edge_v_;
    std::vector<std::size_t> free_slots_;
};

} // namespace algoat::graph::testing

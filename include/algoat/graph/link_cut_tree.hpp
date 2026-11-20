/**
 * @file link_cut_tree.hpp
 * @brief Link-Cut Tree for dynamic forest connectivity and path monoid queries.
 *
 * Implements Sleator and Tarjan's self-adjusting Link-Cut Tree (JCSS 1983 / JACM 1985) using
 * 1-based contiguous flat arrays with index 0 as a dedicated null sentinel, byte-backed lazy
 * reversal tags, and an iterative preallocated splay stack for zero heap allocations after
 * construction.
 */

#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace algoat::graph {

/**
 * @class LinkCutTree
 * @brief Self-adjusting Link-Cut Tree supporting amortized O(log V) forest updates and path
 * queries.
 *
 * Vertices are 0-indexed in [0, size() - 1] externally and mapped to 1-based internal indices
 * [1, size()] with index 0 reserved as an immutable null sentinel. Connectivity, cut, and path
 * queries preserve existing tree roots unless make_root() is invoked explicitly.
 *
 * @tparam ValueType Vertex payload type supporting default construction, copying, operator+, and
 * operator< (must form a commutative monoid for make_root()-invariant path sum/min queries).
 */
template <typename ValueType = int64_t> class LinkCutTree {
    static_assert(std::default_initializable<ValueType>, "ValueType must be default-initializable");
    static_assert(std::copyable<ValueType>, "ValueType must be copyable");
    static_assert(
        requires(const ValueType& a, const ValueType& b) {
            { a + b } -> std::convertible_to<ValueType>;
            { a < b } -> std::convertible_to<bool>;
        }, "ValueType must support binary + and <");

public:
    using value_type = ValueType;

    /**
     * @brief Constructs a forest of @p num_vertices isolated singleton vertices.
     * @throws std::length_error if @p num_vertices equals SIZE_MAX (overflows sentinel slot).
     */
    explicit LinkCutTree(std::size_t num_vertices) {
        if (num_vertices == std::numeric_limits<std::size_t>::max()) {
            throw std::length_error("LinkCutTree: num_vertices exceeds maximum indexable size");
        }
        n_ = num_vertices;
        const std::size_t alloc_size = n_ + 1;
        left_.assign(alloc_size, 0);
        right_.assign(alloc_size, 0);
        parent_.assign(alloc_size, 0);
        reversed_.assign(alloc_size, 0);
        val_.assign(alloc_size, ValueType{});
        path_sum_.assign(alloc_size, ValueType{});
        path_min_.assign(alloc_size, ValueType{});
        splay_stack_.resize(alloc_size);
    }

    /**
     * @brief Links tree root @p u as a child of @p v.
     * @throws std::out_of_range if either vertex index >= size().
     * @throws std::invalid_argument if @p u == @p v, @p u is not a tree root, or @p u and @p v
     * are already connected.
     */
    void link(std::size_t u, std::size_t v) {
        validate_vertex(u);
        validate_vertex(v);
        if (u == v) {
            throw std::invalid_argument("LinkCutTree::link: self-loop is not allowed");
        }
        const std::size_t x = u + 1;
        const std::size_t y = v + 1;
        access(x);
        if (left_[x] != 0) {
            throw std::invalid_argument("LinkCutTree::link: vertex u must be a tree root");
        }
        if (access(y) == x) {
            throw std::invalid_argument(
                "LinkCutTree::link: vertices u and v are already connected");
        }
        parent_[x] = y;
    }

    /**
     * @brief Cuts the direct tree edge connecting @p u and @p v while preserving the upper root.
     * @throws std::out_of_range if either vertex index >= size().
     * @throws std::invalid_argument if @p u == @p v or no direct edge connects @p u and @p v.
     */
    void cut(std::size_t u, std::size_t v) {
        validate_vertex(u);
        validate_vertex(v);
        if (u == v) {
            throw std::invalid_argument("LinkCutTree::cut: self-edge cannot exist");
        }
        const std::size_t x = u + 1;
        const std::size_t y = v + 1;
        if (!try_cut_child(y, x) && !try_cut_child(x, y)) {
            throw std::invalid_argument("LinkCutTree::cut: no direct edge exists between u and v");
        }
    }

    /**
     * @brief Reroots the tree containing @p u at @p u (evert).
     * @throws std::out_of_range if @p u >= size().
     */
    void make_root(std::size_t u) {
        validate_vertex(u);
        const std::size_t x = u + 1;
        access(x);
        apply_rev(x);
    }

    /**
     * @brief Returns true if @p u and @p v belong to the same connected tree without mutating
     * roots.
     * @throws std::out_of_range if either vertex index >= size().
     */
    [[nodiscard]] bool connected(std::size_t u, std::size_t v) {
        validate_vertex(u);
        validate_vertex(v);
        if (u == v) {
            return true;
        }
        const std::size_t x = u + 1;
        const std::size_t y = v + 1;
        access(x);
        const std::size_t lca = access(y);
        return (x == lca) || (parent_[x] != 0);
    }

    /**
     * @brief Returns the root vertex of the tree containing @p u.
     * @throws std::out_of_range if @p u >= size().
     */
    [[nodiscard]] std::size_t find_root(std::size_t u) {
        validate_vertex(u);
        std::size_t x = u + 1;
        access(x);
        push_down(x);
        while (left_[x] != 0) {
            x = left_[x];
            push_down(x);
        }
        // Splay the reached root to pay for the left-spine walk and preserve O(log V) amortized
        // time.
        splay(x);
        return x - 1;
    }

    /**
     * @brief Sets the payload of vertex @p u to @p new_val and updates path aggregates.
     * @throws std::out_of_range if @p u >= size().
     * @throws std::invalid_argument if @p ValueType is floating-point and @p new_val is NaN.
     */
    void update_vertex(std::size_t u, ValueType new_val) {
        validate_vertex(u);
        if constexpr (std::floating_point<ValueType>) {
            if (std::isnan(new_val)) {
                throw std::invalid_argument("LinkCutTree::update_vertex: NaN value is not allowed");
            }
        }
        const std::size_t x = u + 1;
        splay(x);
        val_[x] = std::move(new_val);
        push_up(x);
    }

    /**
     * @brief Returns the sum of vertex values on the unique path between @p u and @p v.
     * @throws std::out_of_range if either vertex index >= size().
     * @throws std::invalid_argument if @p u and @p v are in different trees.
     */
    [[nodiscard]] ValueType query_path_sum(std::size_t u, std::size_t v) {
        return query_path_impl<PathQueryKind::Sum>(u, v);
    }

    /**
     * @brief Returns the minimum vertex value on the unique path between @p u and @p v.
     * @throws std::out_of_range if either vertex index >= size().
     * @throws std::invalid_argument if @p u and @p v are in different trees.
     */
    [[nodiscard]] ValueType query_path_min(std::size_t u, std::size_t v) {
        return query_path_impl<PathQueryKind::Min>(u, v);
    }

    /**
     * @brief Returns the number of vertices in the dynamic forest.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return n_;
    }

private:
    enum class PathQueryKind : uint8_t { Sum, Min };

    void validate_vertex(std::size_t u) const {
        if (u >= n_) {
            throw std::out_of_range("LinkCutTree: vertex index out of bounds");
        }
    }

    [[nodiscard]] bool is_root(std::size_t x) const noexcept {
        const std::size_t p = parent_[x];
        return left_[p] != x && right_[p] != x;
    }

    void apply_rev(std::size_t x) noexcept {
        if (x == 0) {
            return;
        }
        std::swap(left_[x], right_[x]);
        reversed_[x] ^= 1U;
    }

    void push_down(std::size_t x) noexcept {
        if (reversed_[x] != 0) {
            apply_rev(left_[x]);
            apply_rev(right_[x]);
            reversed_[x] = 0;
        }
    }

    void push_up(std::size_t x) {
        ValueType sum = val_[x];
        ValueType mn = val_[x];
        if (const std::size_t l = left_[x]; l != 0) {
            sum = path_sum_[l] + sum;
            if (path_min_[l] < mn) {
                mn = path_min_[l];
            }
        }
        if (const std::size_t r = right_[x]; r != 0) {
            sum = sum + path_sum_[r];
            if (path_min_[r] < mn) {
                mn = path_min_[r];
            }
        }
        path_sum_[x] = std::move(sum);
        path_min_[x] = std::move(mn);
    }

    void rotate(std::size_t x) {
        const std::size_t y = parent_[x];
        const std::size_t z = parent_[y];
        const bool is_right = (right_[y] == x);

        if (!is_root(y)) {
            if (left_[z] == y) {
                left_[z] = x;
            } else {
                right_[z] = x;
            }
        }
        parent_[x] = z;

        const std::size_t b = is_right ? left_[x] : right_[x];
        if (is_right) {
            right_[y] = b;
            left_[x] = y;
        } else {
            left_[y] = b;
            right_[x] = y;
        }
        if (b != 0) {
            parent_[b] = y;
        }
        parent_[y] = x;

        push_up(y);
        push_up(x);
    }

    void splay(std::size_t x) {
        std::size_t top = 0;
        std::size_t curr = x;
        splay_stack_[top++] = curr;
        while (!is_root(curr)) {
            curr = parent_[curr];
            splay_stack_[top++] = curr;
        }
        while (top > 0) {
            push_down(splay_stack_[--top]);
        }

        while (!is_root(x)) {
            const std::size_t y = parent_[x];
            const std::size_t z = parent_[y];
            if (!is_root(y)) {
                const bool zig_zig = (right_[z] == y) == (right_[y] == x);
                rotate(zig_zig ? y : x);
            }
            rotate(x);
        }
    }

    template <bool InvokeOnRoot = false, typename OnRoot>
    std::size_t access_impl(std::size_t x, OnRoot&& on_root) {
        std::size_t last = 0;
        for (std::size_t y = x; y != 0; y = parent_[y]) {
            splay(y);
            if constexpr (InvokeOnRoot) {
                if (parent_[y] == 0) {
                    on_root(y, last);
                }
            }
            right_[y] = last;
            push_up(y);
            last = y;
        }
        splay(x);
        return last;
    }

    std::size_t access(std::size_t x) {
        return access_impl(x, [](std::size_t, std::size_t) noexcept {});
    }

    [[nodiscard]] bool try_cut_child(std::size_t child, std::size_t expected_parent) {
        access(child);
        std::size_t pred = left_[child];
        if (pred == 0) {
            return false;
        }
        push_down(pred);
        while (right_[pred] != 0) {
            pred = right_[pred];
            push_down(pred);
        }
        splay(pred);
        if (pred != expected_parent) {
            return false;
        }
        right_[pred] = 0;
        parent_[child] = 0;
        push_up(pred);
        return true;
    }

    template <PathQueryKind Kind>
    [[nodiscard]] ValueType query_path_impl(std::size_t u, std::size_t v) {
        validate_vertex(u);
        validate_vertex(v);
        const std::size_t x = u + 1;
        const std::size_t y_target = v + 1;
        if (x == y_target) {
            return val_[x];
        }

        access(x);
        bool is_connected = false;
        ValueType result{};
        access_impl<true>(y_target, [&](std::size_t y, std::size_t last) {
            is_connected = (y == x) || (parent_[x] != 0);
            if (!is_connected) {
                return;
            }
            result = val_[y];
            if constexpr (Kind == PathQueryKind::Sum) {
                if (const std::size_t ru = right_[y]; ru != 0) {
                    result = result + path_sum_[ru];
                }
                if (last != 0) {
                    result = result + path_sum_[last];
                }
            } else {
                if (const std::size_t ru = right_[y]; ru != 0 && path_min_[ru] < result) {
                    result = path_min_[ru];
                }
                if (last != 0 && path_min_[last] < result) {
                    result = path_min_[last];
                }
            }
        });

        if (!is_connected) {
            throw std::invalid_argument(
                "LinkCutTree::query_path: vertices u and v are not connected");
        }
        return result;
    }

    std::size_t n_{0};

    std::vector<std::size_t> left_;
    std::vector<std::size_t> right_;
    std::vector<std::size_t> parent_;
    std::vector<uint8_t> reversed_;
    std::vector<ValueType> val_;
    std::vector<ValueType> path_sum_;
    std::vector<ValueType> path_min_;
    std::vector<std::size_t> splay_stack_;
};

} // namespace algoat::graph

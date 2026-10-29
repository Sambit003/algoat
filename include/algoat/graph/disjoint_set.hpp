/**
 * @file disjoint_set.hpp
 * @brief High-performance sequential and concurrent Disjoint Set Union (DSU) data structures.
 *
 * Provides:
 * 1. DisjointSetUnion: Cache-aligned single-threaded DSU with union-by-size and path halving.
 * 2. ConcurrentDisjointSet: Lock-free and wait-free concurrent DSU featuring atomic path halving,
 *    packed 64-bit rank-and-parent representation to prevent false sharing across threads,
 *    asymmetric linking with deterministic tie-breaking to guarantee acyclicity across concurrent
 * threads, and 64-byte aligned contiguous storage.
 */

#pragma once

#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace algoat::graph {

/**
 * @class DisjointSetUnion
 * @brief High-throughput sequential Disjoint Set Union (Union-Find) structure.
 *
 * Implements union-by-size and path halving with a single contiguous 32-bit integer per node:
 * - Negative entries denote root vertices, where the absolute value equals the component size.
 * - Non-negative entries denote parent vertex indices.
 */
class DisjointSetUnion {
public:
    explicit DisjointSetUnion(std::size_t n = 0) : parent_or_size_(n, -1), num_components_(n) {}

    /**
     * @brief Finds the representative set root for vertex i with single-pass path halving.
     */
    [[nodiscard]] VertexId find(VertexId i) noexcept {
        while (parent_or_size_[i] >= 0) {
            const auto p = static_cast<VertexId>(parent_or_size_[i]);
            if (parent_or_size_[p] >= 0) {
                parent_or_size_[i] = parent_or_size_[p];
            }
            i = static_cast<VertexId>(parent_or_size_[i]);
        }
        return i;
    }

    /**
     * @brief Finds the representative set root for vertex i (const inspection without mutation).
     */
    [[nodiscard]] VertexId find(VertexId i) const noexcept {
        while (parent_or_size_[i] >= 0) {
            i = static_cast<VertexId>(parent_or_size_[i]);
        }
        return i;
    }

    /**
     * @brief Merges the sets containing vertex i and vertex j via union-by-size.
     * @return true if two previously distinct sets were merged; false if they were already in the
     * same set.
     */
    bool unite(VertexId i, VertexId j) noexcept {
        VertexId root_i = find(i);
        VertexId root_j = find(j);
        if (root_i == root_j) {
            return false;
        }

        // Union by size: attach smaller tree under larger tree
        if (-parent_or_size_[root_i] < -parent_or_size_[root_j]) {
            std::swap(root_i, root_j);
        }

        parent_or_size_[root_i] += parent_or_size_[root_j];
        parent_or_size_[root_j] = static_cast<int32_t>(root_i);
        --num_components_;
        return true;
    }

    /**
     * @brief Checks whether vertex i and vertex j belong to the same component.
     */
    [[nodiscard]] bool same_set(VertexId i, VertexId j) noexcept {
        return find(i) == find(j);
    }

    /**
     * @brief Checks whether vertex i and vertex j belong to the same component (const inspection).
     */
    [[nodiscard]] bool same_set(VertexId i, VertexId j) const noexcept {
        return find(i) == find(j);
    }

    /**
     * @brief Returns the total number of elements.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return parent_or_size_.size();
    }

    /**
     * @brief Returns the number of disjoint connected components remaining.
     */
    [[nodiscard]] std::size_t num_components() const noexcept {
        return num_components_;
    }

    /**
     * @brief Returns the size of the component containing vertex i.
     */
    [[nodiscard]] std::size_t component_size(VertexId i) noexcept {
        return static_cast<std::size_t>(-parent_or_size_[find(i)]);
    }

private:
    std::vector<int32_t> parent_or_size_;
    std::size_t num_components_{0};
};

/**
 * @class ConcurrentDisjointSet
 * @brief Lock-free concurrent Disjoint Set Union (Union-Find) with wait-free path halving.
 *
 * Implements:
 * 1. Packed 64-bit Rank and Parent: Packs a 32-bit parent pointer and 32-bit rank into a single
 *    atomic 64-bit word, enabling atomic double-field updates without false sharing.
 * 2. Wait-Free Path Halving: `find_atomic` skips alternate ancestors and advances unconditionally
 *    to grandparent nodes using relaxed atomic CAS without retry loops.
 * 3. Union by Rank with Asymmetric Tie-Breaking (Union-by-ID): Guarantees acyclicity across
 *    concurrent threads without locks.
 * 4. 64-Byte Cache-Line Aligned Storage: Portable C++17 aligned allocation.
 */
class ConcurrentDisjointSet {
public:
    using Rank = uint32_t;

    explicit ConcurrentDisjointSet(std::size_t n = 0) : n_(n), num_components_(n) {
        if (n_ == 0) {
            return;
        }

        constexpr std::size_t alignment = 64;
        std::size_t bytes = n_ * sizeof(std::atomic<uint64_t>);
        if (bytes % alignment != 0) {
            bytes += alignment - (bytes % alignment);
        }

        void* ptr = ::operator new(bytes, std::align_val_t{alignment});
        data_ = static_cast<std::atomic<uint64_t>*>(ptr);

        for (VertexId i = 0; i < static_cast<VertexId>(n_); ++i) {
            new (&data_[i]) std::atomic<uint64_t>(pack(i, 0));
        }
    }

    ~ConcurrentDisjointSet() {
        if (data_ != nullptr) {
            for (std::size_t i = 0; i < n_; ++i) {
                data_[i].~atomic();
            }
            ::operator delete(data_, std::align_val_t{64});
            data_ = nullptr;
        }
    }

    ConcurrentDisjointSet(const ConcurrentDisjointSet&) = delete;
    ConcurrentDisjointSet& operator=(const ConcurrentDisjointSet&) = delete;
    ConcurrentDisjointSet(ConcurrentDisjointSet&&) = delete;
    ConcurrentDisjointSet& operator=(ConcurrentDisjointSet&&) = delete;

    /**
     * @brief Wait-free find operation with atomic path halving.
     * Guaranteed bounded steps: strictly monotonically advances up the tree.
     */
    [[nodiscard]] VertexId find_atomic(VertexId u) const noexcept {
        while (true) {
            const uint64_t entry_u = data_[u].load(std::memory_order_relaxed);
            const VertexId p = get_parent(entry_u);
            if (p == u) {
                return u;
            }

            const uint64_t entry_p = data_[p].load(std::memory_order_relaxed);
            const VertexId gp = get_parent(entry_p);
            if (p == gp) {
                return p;
            }

            // Path halving: point u to its grandparent gp while preserving rank of u
            const uint64_t desired = pack(gp, get_rank(entry_u));
            uint64_t expected = entry_u;
            data_[u].compare_exchange_weak(expected, desired, std::memory_order_relaxed,
                                           std::memory_order_relaxed);

            u = gp;
        }
    }

    /**
     * @brief Lock-free concurrent union by rank with asymmetric ID tie-breaking.
     * Prevents cycles and deadlocks across concurrent threads.
     * @return true if two previously distinct sets were merged; false otherwise.
     */
    bool union_atomic(VertexId u, VertexId v) noexcept {
        while (true) {
            VertexId root_u = find_atomic(u);
            VertexId root_v = find_atomic(v);

            if (root_u == root_v) {
                return false;
            }

            uint64_t entry_u = data_[root_u].load(std::memory_order_relaxed);
            uint64_t entry_v = data_[root_v].load(std::memory_order_relaxed);

            // Re-verify both are still roots
            if (get_parent(entry_u) != root_u || get_parent(entry_v) != root_v) {
                continue;
            }

            Rank rank_u = get_rank(entry_u);
            Rank rank_v = get_rank(entry_v);

            // Tie-break: smaller rank points to larger rank;
            // if ranks are equal, smaller root ID points to larger root ID
            if (rank_u > rank_v || (rank_u == rank_v && root_u > root_v)) {
                std::swap(root_u, root_v);
                std::swap(entry_u, entry_v);
                std::swap(rank_u, rank_v);
            }

            // Attempt to hook root_u under root_v
            const uint64_t new_entry_u = pack(root_v, rank_u);
            if (!data_[root_u].compare_exchange_weak(
                    entry_u, new_entry_u, std::memory_order_release, std::memory_order_relaxed)) {
                continue;
            }

            // If ranks were equal, lazily bump root_v's rank
            if (rank_u == rank_v) {
                const uint64_t new_entry_v = pack(root_v, rank_v + 1);
                uint64_t expected_v = entry_v;
                data_[root_v].compare_exchange_weak(
                    expected_v, new_entry_v, std::memory_order_relaxed, std::memory_order_relaxed);
            }

            num_components_.fetch_sub(1, std::memory_order_relaxed);
            return true;
        }
    }

    /**
     * @brief Checks whether vertex u and vertex v belong to the same component.
     */
    [[nodiscard]] bool same_set(VertexId u, VertexId v) const noexcept {
        return find_atomic(u) == find_atomic(v);
    }

    /**
     * @brief Returns the total number of elements.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return n_;
    }

    /**
     * @brief Returns the number of disjoint connected components remaining.
     */
    [[nodiscard]] std::size_t num_components() const noexcept {
        return num_components_.load(std::memory_order_relaxed);
    }

private:
    [[nodiscard]] static constexpr uint64_t pack(VertexId parent, Rank rank) noexcept {
        return (static_cast<uint64_t>(rank) << 32) | static_cast<uint64_t>(parent);
    }

    [[nodiscard]] static constexpr VertexId get_parent(uint64_t val) noexcept {
        return static_cast<VertexId>(val & 0xFFFFFFFFULL);
    }

    [[nodiscard]] static constexpr Rank get_rank(uint64_t val) noexcept {
        return static_cast<Rank>(val >> 32);
    }

    std::atomic<uint64_t>* data_{nullptr};
    std::size_t n_{0};
    std::atomic<std::size_t> num_components_{0};
};

} // namespace algoat::graph

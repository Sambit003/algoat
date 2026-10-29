/**
 * @file mst.hpp
 * @brief Scalable Parallel and Serial Minimum Spanning Tree / Spanning Forest Engine.
 *
 * Implements:
 * 1. Hybrid Parallel Boruvka-Kruskal Minimum Spanning Forest Engine (Bader & Cong 2006).
 * 2. High-Performance Single-Threaded Kruskal Engine with Cache-Aligned Disjoint Set Union.
 * 3. Wait-Free and Lock-Free Path-Halving DSU integration.
 * 4. Canonical strict total ordering for deterministic tie-breaking and cycle prevention.
 */

#pragma once

#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/disjoint_set.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace algoat::graph {

/**
 * @struct MSTEdge
 * @brief Representation of an undirected edge in a Minimum Spanning Tree / Spanning Forest.
 */
template <typename WeightType = double> struct MSTEdge {
    VertexId u{0};
    VertexId v{0};
    WeightType weight{0};

    [[nodiscard]] auto canonical_key() const noexcept {
        return std::tuple(weight, std::min(u, v), std::max(u, v));
    }

    bool operator<(const MSTEdge& other) const noexcept {
        if constexpr (std::floating_point<WeightType>) {
            if (std::isnan(weight) || std::isnan(other.weight)) {
                return !std::isnan(weight) && std::isnan(other.weight);
            }
        }
        return canonical_key() < other.canonical_key();
    }

    bool operator==(const MSTEdge& other) const noexcept {
        return std::min(u, v) == std::min(other.u, other.v) &&
               std::max(u, v) == std::max(other.u, other.v) && weight == other.weight;
    }
};

/**
 * @struct MSTResult
 * @brief Output container for Minimum Spanning Tree (MST) and Spanning Forest (MSF) solutions.
 */
template <typename WeightType = double> struct MSTResult {
    WeightType total_weight{0};
    std::vector<MSTEdge<WeightType>> edges;
    std::size_t components_count{0};
};

namespace detail {

/**
 * @struct SpinLock
 * @brief 64-byte cache-line aligned acquire-release spinlock for striping component edge updates.
 */
struct alignas(64) SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

    void lock() noexcept {
        while (flag.test_and_set(std::memory_order_acquire)) {
#if defined(_MSC_VER)
#if defined(_M_X64) || defined(_M_IX86)
            _mm_pause();
#elif defined(_M_ARM64)
            __yield();
#endif
#elif defined(__x86_64__) || defined(__i386__)
            _mm_pause();
#elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
#endif
        }
    }

    void unlock() noexcept {
        flag.clear(std::memory_order_release);
    }
};

} // namespace detail

/**
 * @class MinimumSpanningTreeEngine
 * @brief Scalable state-of-the-art Minimum Spanning Tree / Forest engine for CSRGraph.
 */
template <typename WeightType = double> class MinimumSpanningTreeEngine {
public:
    explicit MinimumSpanningTreeEngine(const CSRGraph<WeightType>& graph) : graph_(graph) {}

    /**
     * @brief High-throughput single-threaded Kruskal algorithm with cache-aligned DSU.
     */
    [[nodiscard]] MSTResult<WeightType> compute_serial() const {
        const std::size_t num_v = graph_.num_vertices();
        MSTResult<WeightType> result;

        if (num_v == 0) {
            result.components_count = 0;
            return result;
        }
        if (num_v == 1) {
            result.components_count = 1;
            return result;
        }

        std::vector<MSTEdge<WeightType>> candidate_edges;
        candidate_edges.reserve(graph_.num_edges() / 2);

        for (VertexId u = 0; u < num_v; ++u) {
            auto neighbors = graph_.out_neighbors(u);
            auto weights = graph_.out_weights(u);
            const std::size_t deg = neighbors.size();

            for (std::size_t i = 0; i < deg; ++i) {
                const VertexId v = neighbors[i];
                const WeightType w = weights[i];

                if constexpr (std::floating_point<WeightType>) {
                    if (std::isnan(w)) {
                        throw std::invalid_argument("NaN edge weight detected in MST computation");
                    }
                }

                if (u == v) {
                    continue; // Skip self-loops
                }

                if (!graph_.is_directed() && u > v) {
                    continue; // Deduplicate reverse undirected arcs
                }

                candidate_edges.push_back(MSTEdge<WeightType>{u, v, w});
            }
        }

        std::sort(candidate_edges.begin(), candidate_edges.end());

        DisjointSetUnion dsu(num_v);
        result.edges.reserve(num_v - 1);

        for (const auto& edge : candidate_edges) {
            if (dsu.unite(edge.u, edge.v)) {
                result.edges.push_back(edge);
                if (result.edges.size() + 1 == num_v) {
                    break;
                }
            }
        }

        result.components_count = dsu.num_components();
        std::sort(result.edges.begin(), result.edges.end());

        WeightType total = 0;
        for (const auto& e : result.edges) {
            total += e.weight;
        }
        result.total_weight = total;

        return result;
    }

    /**
     * @brief Hybrid Parallel Boruvka-Kruskal Minimum Spanning Forest engine (Bader & Cong 2006).
     *
     * Combines parallel Boruvka contraction phases with wait-free path halving and packed
     * rank-and-parent DSU, transitioning to Kruskal when the contracted graph fits within cache.
     *
     * @param num_threads Number of worker threads (0 selects default system threads).
     */
    [[nodiscard]] MSTResult<WeightType> compute_parallel(std::size_t num_threads = 0) const {
        const std::size_t num_v = graph_.num_vertices();
        MSTResult<WeightType> result;

        if (num_v == 0) {
            result.components_count = 0;
            return result;
        }
        if (num_v == 1) {
            result.components_count = 1;
            return result;
        }

        int threads = static_cast<int>(num_threads);
#if defined(_OPENMP)
        if (threads <= 0) {
            threads = omp_get_max_threads();
        }
#else
        threads = 1;
#endif
        if (threads <= 0) {
            threads = 1;
        }

        // Validate against NaN edge weights upfront
        const auto signed_num_v = static_cast<std::ptrdiff_t>(num_v);
        std::atomic<bool> has_nan{false};

#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
        for (std::ptrdiff_t u = 0; u < signed_num_v; ++u) {
            if (has_nan.load(std::memory_order_relaxed)) {
                continue;
            }
            auto weights = graph_.out_weights(static_cast<VertexId>(u));
            for (const auto& w : weights) {
                if constexpr (std::floating_point<WeightType>) {
                    if (std::isnan(w)) {
                        has_nan.store(true, std::memory_order_relaxed);
                    }
                }
            }
        }

        if (has_nan.load(std::memory_order_relaxed)) {
            throw std::invalid_argument("NaN edge weight detected in MST computation");
        }

        ConcurrentDisjointSet cdsu(num_v);
        std::vector<MSTEdge<WeightType>> result_edges;
        result_edges.reserve(num_v - 1);

        // Pre-allocate 64-byte aligned spinlock striping buffer (4096 entries)
        constexpr std::size_t NUM_LOCKS = 4096;
        static_assert((NUM_LOCKS & (NUM_LOCKS - 1)) == 0, "NUM_LOCKS must be power of two");
        std::vector<detail::SpinLock> lock_stripes(NUM_LOCKS);

        std::vector<MSTEdge<WeightType>> min_edges(num_v);
        std::vector<uint8_t> has_min_edge(num_v, 0);

        // Threshold to transition from parallel Boruvka contraction to sequential Kruskal on
        // super-graph
        const std::size_t base_threshold =
            std::max<std::size_t>(1024, static_cast<std::size_t>(threads) * 64);

        // Phase 1: Parallel Boruvka Contraction Rounds (Bader & Cong 2006)
        while (cdsu.num_components() > base_threshold) {
            // Reset component cheapest edge records
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
            for (std::ptrdiff_t u = 0; u < signed_num_v; ++u) {
                has_min_edge[static_cast<std::size_t>(u)] = 0;
            }

            // Step 1: Concurrently scan edges to identify the minimum incident cross-edge per
            // component
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1024)
#endif
            for (std::ptrdiff_t u_idx = 0; u_idx < signed_num_v; ++u_idx) {
                const auto u = static_cast<VertexId>(u_idx);
                const VertexId comp_u = cdsu.find_atomic(u);

                auto neighbors = graph_.out_neighbors(u);
                auto weights = graph_.out_weights(u);
                const std::size_t deg = neighbors.size();

                for (std::size_t i = 0; i < deg; ++i) {
                    const VertexId v = neighbors[i];
                    if (u == v) {
                        continue;
                    }

                    const VertexId comp_v = cdsu.find_atomic(v);
                    if (comp_u == comp_v) {
                        continue; // Same super-vertex
                    }

                    const WeightType w = weights[i];

                    // Fast lock-free pre-check: skip if current best edge is already strictly
                    // lighter
                    if (has_min_edge[comp_u] && w > min_edges[comp_u].weight) {
                        continue;
                    }

                    const MSTEdge<WeightType> candidate{u, v, w};

                    const std::size_t lock_idx = static_cast<std::size_t>(comp_u) & (NUM_LOCKS - 1);
                    lock_stripes[lock_idx].lock();
                    if (!has_min_edge[comp_u] || candidate < min_edges[comp_u]) {
                        min_edges[comp_u] = candidate;
                        has_min_edge[comp_u] = 1;
                    }
                    lock_stripes[lock_idx].unlock();
                }
            }

            // Step 2: Concurrently unite components along selected minimum edges
            std::vector<std::vector<MSTEdge<WeightType>>> thread_edges(threads);

#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
            {
                int tid = 0;
#if defined(_OPENMP)
                tid = omp_get_thread_num();
#endif
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 1024)
#endif
                for (std::ptrdiff_t c = 0; c < signed_num_v; ++c) {
                    const auto comp_idx = static_cast<std::size_t>(c);
                    if (has_min_edge[comp_idx]) {
                        const auto& edge = min_edges[comp_idx];
                        if (cdsu.union_atomic(edge.u, edge.v)) {
                            thread_edges[static_cast<std::size_t>(tid)].push_back(edge);
                        }
                    }
                }
            }

            std::size_t edges_added_this_round = 0;
            for (const auto& te : thread_edges) {
                edges_added_this_round += te.size();
                result_edges.insert(result_edges.end(), te.begin(), te.end());
            }

            // If no edges could be added, remaining components are disconnected
            if (edges_added_this_round == 0) {
                break;
            }
        }

        // Phase 2: Kruskal on Super-Graph (when graph fits inside L3 cache)
        if (cdsu.num_components() > 1) {
            std::vector<MSTEdge<WeightType>> remaining_edges;
            std::vector<std::vector<MSTEdge<WeightType>>> thread_remaining(threads);

#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
            {
                int tid = 0;
#if defined(_OPENMP)
                tid = omp_get_thread_num();
#endif
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 1024)
#endif
                for (std::ptrdiff_t u_idx = 0; u_idx < signed_num_v; ++u_idx) {
                    const auto u = static_cast<VertexId>(u_idx);
                    const VertexId comp_u = cdsu.find_atomic(u);

                    auto neighbors = graph_.out_neighbors(u);
                    auto weights = graph_.out_weights(u);
                    const std::size_t deg = neighbors.size();

                    for (std::size_t i = 0; i < deg; ++i) {
                        const VertexId v = neighbors[i];
                        if (u >= v) {
                            continue; // Sift unique undirected edges
                        }

                        const VertexId comp_v = cdsu.find_atomic(v);
                        if (comp_u == comp_v) {
                            continue; // Already connected
                        }

                        const WeightType w = weights[i];
                        thread_remaining[static_cast<std::size_t>(tid)].push_back(
                            MSTEdge<WeightType>{u, v, w});
                    }
                }
            }

            for (auto& te : thread_remaining) {
                remaining_edges.insert(remaining_edges.end(), te.begin(), te.end());
            }

            std::sort(remaining_edges.begin(), remaining_edges.end());

            for (const auto& edge : remaining_edges) {
                if (cdsu.union_atomic(edge.u, edge.v)) {
                    result_edges.push_back(edge);
                    if (cdsu.num_components() <= 1) {
                        break;
                    }
                }
            }
        }

        result.components_count = cdsu.num_components();
        std::sort(result_edges.begin(), result_edges.end());

        WeightType total = 0;
        for (const auto& e : result_edges) {
            total += e.weight;
        }
        result.total_weight = total;
        result.edges = std::move(result_edges);

        return result;
    }

private:
    const CSRGraph<WeightType>& graph_;
};

} // namespace algoat::graph

/**
 * @file centrality.hpp
 * @brief High-throughput network centrality engine (PageRank & Brandes Betweenness Centrality).
 *
 * Implements:
 * 1. Vectorized SpMV PageRank with exact dangling node mass redistribution, double-buffering,
 *    and AVX2/AVX-512 FMA acceleration.
 * 2. Brandes' Betweenness Centrality (2001) in O(V * E) time and O(V + E) auxiliary space.
 * 3. Freeman (1978) normalized Degree Centrality bounded in [0, 1].
 */

#pragma once

#include "algoat/graph/csr_graph.hpp"
#include "algoat/graph/d_ary_heap.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#endif

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <type_traits>
#include <vector>

namespace algoat::graph {

/**
 * @struct PageRankConfig
 * @brief Configuration parameters for PageRank power iteration.
 */
struct PageRankConfig {
    double damping_factor{0.85};
    double tolerance{1e-6};
    std::size_t max_iterations{100};
};

/**
 * @struct PageRankResult
 * @brief Output container for PageRank computation.
 */
struct PageRankResult {
    std::vector<double> scores;
    std::size_t iterations_converged{0};
    double final_l1_delta{0.0};
};

/**
 * @struct CentralityResult
 * @brief Output container for Betweenness and Degree Centrality metrics.
 */
struct CentralityResult {
    std::vector<double> betweenness;
    std::vector<double> degree_centrality;
};

/**
 * @brief Computes normalized Freeman Degree Centrality bounded in [0, 1].
 *
 * For undirected graphs: deg(v) / (N - 1).
 * For directed graphs: (in_deg(v) + out_deg(v)) / (2 * (N - 1)).
 * Self-loops are excluded per Freeman (1978). For N <= 1, returns zeros.
 *
 * @param graph Input CSR graph.
 * @return std::vector<double> Degree centrality scores for each vertex.
 */
template <typename WeightType = void>
[[nodiscard]] std::vector<double> degree_centrality(const CSRGraph<WeightType>& graph) {
    const std::size_t num_v = graph.num_vertices();
    if (num_v <= 1) {
        return std::vector<double>(num_v, 0.0);
    }

    std::vector<double> result(num_v, 0.0);
    const double norm_factor = 1.0 / static_cast<double>(num_v - 1);

    for (std::size_t v = 0; v < num_v; ++v) {
        const auto vid = static_cast<VertexId>(v);
        auto out_neigh = graph.out_neighbors(vid);
        const bool has_self_loop = std::binary_search(out_neigh.begin(), out_neigh.end(), vid);
        const std::size_t self_sub = has_self_loop ? 1 : 0;

        if (!graph.is_directed()) {
            const std::size_t deg = out_neigh.size() - self_sub;
            result[v] = static_cast<double>(deg) * norm_factor;
        } else {
            auto in_neigh = graph.in_neighbors(vid);
            const std::size_t in_deg = in_neigh.size() - self_sub;
            const std::size_t out_deg = out_neigh.size() - self_sub;
            result[v] = static_cast<double>(in_deg + out_deg) * (norm_factor * 0.5);
        }
    }
    return result;
}

/**
 * @brief Computes PageRank scores across graph vertices using pull-direction SpMV.
 *
 * Guaranteed to satisfy the total probability sum invariant (sum(p_i) == 1.0) by
 * redistributing dangling node mass uniformly on each iteration.
 *
 * @param graph Input CSR graph (directed or undirected).
 * @param config PageRank configuration parameters (damping factor, tolerance, max iterations).
 * @return PageRankResult containing score vector and convergence statistics.
 */
template <typename WeightType = void>
[[nodiscard]] PageRankResult pagerank(const CSRGraph<WeightType>& graph,
                                      PageRankConfig config = {}) {
    const std::size_t num_v = graph.num_vertices();
    if (num_v == 0) {
        return PageRankResult{.scores = {}, .iterations_converged = 0, .final_l1_delta = 0.0};
    }
    if (num_v == 1) {
        return PageRankResult{.scores = {1.0}, .iterations_converged = 1, .final_l1_delta = 0.0};
    }

    const double alpha = config.damping_factor;
    const auto signed_num_v = static_cast<std::ptrdiff_t>(num_v);

    // Precompute reciprocal out-degrees and locate dangling nodes
    std::vector<double> inv_out_deg(num_v, 0.0);
    std::vector<VertexId> dangling_nodes;
    dangling_nodes.reserve(num_v / 8);

    for (VertexId u = 0; u < num_v; ++u) {
        auto deg = graph.out_degree(u);
        if (deg > 0) {
            inv_out_deg[u] = 1.0 / static_cast<double>(deg);
        } else {
            dangling_nodes.push_back(u);
        }
    }

    // Double-buffered score vectors
    std::vector<double> p_curr(num_v, 1.0 / static_cast<double>(num_v));
    std::vector<double> p_next(num_v, 0.0);

    std::size_t iters_converged = config.max_iterations;
    double last_delta = 0.0;

    for (std::size_t iter = 0; iter < config.max_iterations; ++iter) {
        // Accumulate dangling node mass
        double dangling_mass = 0.0;
        const auto num_dangling = static_cast<std::ptrdiff_t>(dangling_nodes.size());

#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : dangling_mass) schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < num_dangling; ++i) {
            dangling_mass += p_curr[dangling_nodes[static_cast<std::size_t>(i)]];
        }

        const double base_score =
            (1.0 - alpha + alpha * dangling_mass) / static_cast<double>(num_v);

        // Pull-direction SpMV over incoming neighbors
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for (std::ptrdiff_t v_idx = 0; v_idx < signed_num_v; ++v_idx) {
            const auto v = static_cast<VertexId>(v_idx);
            auto in_neigh = graph.in_neighbors(v);
            const std::size_t in_deg = in_neigh.size();
            const VertexId* in_ptr = in_neigh.data();
            double incoming_sum = 0.0;

#if defined(__AVX512F__)
            std::size_t i = 0;
            if (in_deg >= 8) {
                __m512d acc = _mm512_setzero_pd();
                for (; i + 8 <= in_deg; i += 8) {
                    const __m512d p_val = _mm512_set_pd(
                        p_curr[in_ptr[i + 7]], p_curr[in_ptr[i + 6]], p_curr[in_ptr[i + 5]],
                        p_curr[in_ptr[i + 4]], p_curr[in_ptr[i + 3]], p_curr[in_ptr[i + 2]],
                        p_curr[in_ptr[i + 1]], p_curr[in_ptr[i + 0]]);
                    const __m512d inv_d =
                        _mm512_set_pd(inv_out_deg[in_ptr[i + 7]], inv_out_deg[in_ptr[i + 6]],
                                      inv_out_deg[in_ptr[i + 5]], inv_out_deg[in_ptr[i + 4]],
                                      inv_out_deg[in_ptr[i + 3]], inv_out_deg[in_ptr[i + 2]],
                                      inv_out_deg[in_ptr[i + 1]], inv_out_deg[in_ptr[i + 0]]);
                    acc = _mm512_fmadd_pd(p_val, inv_d, acc);
                }
                incoming_sum += _mm512_reduce_add_pd(acc);
            }
            for (; i < in_deg; ++i) {
                const VertexId u = in_ptr[i];
                incoming_sum += p_curr[u] * inv_out_deg[u];
            }
#elif defined(__AVX2__) && defined(__FMA__)
            std::size_t i = 0;
            if (in_deg >= 4) {
                __m256d acc = _mm256_setzero_pd();
                for (; i + 4 <= in_deg; i += 4) {
                    const VertexId u0 = in_ptr[i];
                    const VertexId u1 = in_ptr[i + 1];
                    const VertexId u2 = in_ptr[i + 2];
                    const VertexId u3 = in_ptr[i + 3];
                    const __m256d p_val =
                        _mm256_set_pd(p_curr[u3], p_curr[u2], p_curr[u1], p_curr[u0]);
                    const __m256d inv_d = _mm256_set_pd(inv_out_deg[u3], inv_out_deg[u2],
                                                        inv_out_deg[u1], inv_out_deg[u0]);
                    acc = _mm256_fmadd_pd(p_val, inv_d, acc);
                }
                alignas(32) double temp[4];
                _mm256_storeu_pd(temp, acc);
                incoming_sum += temp[0] + temp[1] + temp[2] + temp[3];
            }
            for (; i < in_deg; ++i) {
                const VertexId u = in_ptr[i];
                incoming_sum += p_curr[u] * inv_out_deg[u];
            }
#else
            for (std::size_t i = 0; i < in_deg; ++i) {
                const VertexId u = in_ptr[i];
                incoming_sum += p_curr[u] * inv_out_deg[u];
            }
#endif

            p_next[v] = base_score + alpha * incoming_sum;
        }

        // Vectorized L1 delta calculation using _mm256_sub_pd and _mm256_andnot_pd
        double delta = 0.0;
#if defined(__AVX2__)
        std::size_t v_idx = 0;
        __m256d delta_acc = _mm256_setzero_pd();
        // Sign bit mask: bit 63 set, clearing sign bit computes absolute value
        const __m256d sign_mask =
            _mm256_castsi256_pd(_mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
        for (; v_idx + 4 <= num_v; v_idx += 4) {
            const __m256d cur = _mm256_loadu_pd(p_curr.data() + v_idx);
            const __m256d nxt = _mm256_loadu_pd(p_next.data() + v_idx);
            const __m256d diff = _mm256_sub_pd(nxt, cur);
            const __m256d abs_diff = _mm256_andnot_pd(sign_mask, diff);
            delta_acc = _mm256_add_pd(delta_acc, abs_diff);
        }
        alignas(32) double dtemp[4];
        _mm256_storeu_pd(dtemp, delta_acc);
        delta = dtemp[0] + dtemp[1] + dtemp[2] + dtemp[3];
        for (; v_idx < num_v; ++v_idx) {
            delta += std::abs(p_next[v_idx] - p_curr[v_idx]);
        }
#else
        for (std::size_t v = 0; v < num_v; ++v) {
            delta += std::abs(p_next[v] - p_curr[v]);
        }
#endif

        last_delta = delta;
        p_curr.swap(p_next);

        if (delta < config.tolerance) {
            iters_converged = iter + 1;
            break;
        }
    }

    // Preserve exact probability sum invariant sum(p_i) == 1.0 against floating-point drift
    const double sum = std::accumulate(p_curr.begin(), p_curr.end(), 0.0);
    if (std::abs(sum - 1.0) > 1e-12 && sum > 0.0) {
        const double inv_sum = 1.0 / sum;
        for (auto& s : p_curr) {
            s *= inv_sum;
        }
    }

    return PageRankResult{
        .scores = std::move(p_curr),
        .iterations_converged = iters_converged,
        .final_l1_delta = last_delta,
    };
}

namespace detail {

/**
 * @struct BrandesWorkspace
 * @brief Reusable thread-local buffer pool for Brandes dependency accumulation.
 */
struct BrandesWorkspace {
    std::vector<std::vector<VertexId>> pred;
    std::vector<double> sigma;
    std::vector<double> delta;
    std::vector<VertexId> stack;

    void init(std::size_t num_v) {
        pred.resize(num_v);
        sigma.resize(num_v, 0.0);
        delta.resize(num_v, 0.0);
        stack.reserve(num_v);
    }
};

template <typename WeightType>
void accumulate_source_unweighted(VertexId s, const CSRGraph<WeightType>& graph,
                                  BrandesWorkspace& ws, std::vector<int32_t>& dist,
                                  std::queue<VertexId>& queue,
                                  std::vector<double>& out_betweenness) {
    const std::size_t num_v = graph.num_vertices();
    for (std::size_t w = 0; w < num_v; ++w) {
        ws.pred[w].clear();
        ws.sigma[w] = 0.0;
        dist[w] = -1;
        ws.delta[w] = 0.0;
    }
    ws.stack.clear();

    dist[s] = 0;
    ws.sigma[s] = 1.0;
    queue.push(s);

    while (!queue.empty()) {
        const VertexId v = queue.front();
        queue.pop();
        ws.stack.push_back(v);

        for (VertexId w : graph.out_neighbors(v)) {
            if (dist[w] < 0) {
                dist[w] = dist[v] + 1;
                queue.push(w);
            }
            if (dist[w] == dist[v] + 1) {
                ws.sigma[w] += ws.sigma[v];
                ws.pred[w].push_back(v);
            }
        }
    }

    while (!ws.stack.empty()) {
        const VertexId w = ws.stack.back();
        ws.stack.pop_back();

        for (VertexId v : ws.pred[w]) {
            ws.delta[v] += (ws.sigma[v] / ws.sigma[w]) * (1.0 + ws.delta[w]);
        }
        if (w != s) {
            out_betweenness[w] += ws.delta[w];
        }
    }
}

template <typename WeightType>
void accumulate_source_weighted(VertexId s, const CSRGraph<WeightType>& graph, BrandesWorkspace& ws,
                                std::vector<WeightType>& dist,
                                d_ary_heap<4, WeightType, VertexId>& heap,
                                std::vector<double>& out_betweenness) {
    const std::size_t num_v = graph.num_vertices();
    const WeightType inf_dist = std::numeric_limits<WeightType>::infinity();
    constexpr WeightType eps = static_cast<WeightType>(1e-9);

    for (std::size_t w = 0; w < num_v; ++w) {
        ws.pred[w].clear();
        ws.sigma[w] = 0.0;
        dist[w] = inf_dist;
        ws.delta[w] = 0.0;
    }
    ws.stack.clear();

    dist[s] = WeightType{0};
    ws.sigma[s] = 1.0;
    heap.push(s, WeightType{0});

    while (!heap.empty()) {
        auto [v, d_v] = heap.pop();
        if (d_v > dist[v]) {
            continue;
        }
        ws.stack.push_back(v);

        auto neighbors = graph.out_neighbors(v);
        auto weights = graph.out_weights(v);
        const std::size_t deg = neighbors.size();

        for (std::size_t i = 0; i < deg; ++i) {
            const VertexId w = neighbors[i];
            const WeightType wt = weights[i];
            const WeightType new_dist = d_v + wt;

            if (new_dist < dist[w] - eps) {
                dist[w] = new_dist;
                ws.sigma[w] = ws.sigma[v];
                ws.pred[w].clear();
                ws.pred[w].push_back(v);
                heap.push(w, new_dist);
            } else if (std::abs(new_dist - dist[w]) <= eps) {
                ws.sigma[w] += ws.sigma[v];
                ws.pred[w].push_back(v);
            }
        }
    }

    while (!ws.stack.empty()) {
        const VertexId w = ws.stack.back();
        ws.stack.pop_back();

        for (VertexId v : ws.pred[w]) {
            ws.delta[v] += (ws.sigma[v] / ws.sigma[w]) * (1.0 + ws.delta[w]);
        }
        if (w != s) {
            out_betweenness[w] += ws.delta[w];
        }
    }
}

} // namespace detail

/**
 * @brief Computes exact Brandes Betweenness Centrality and Degree Centrality in O(V * E) time.
 *
 * Accumulates dependencies in reverse topological order from SSSP exploration trees.
 * For undirected graphs, betweenness scores are divided by 2 to account for path symmetry.
 *
 * @param graph Input CSR graph (directed or undirected, unweighted or weighted).
 * @return CentralityResult containing betweenness and degree centrality arrays.
 */
template <typename WeightType = void>
[[nodiscard]] CentralityResult betweenness_centrality(const CSRGraph<WeightType>& graph) {
    const std::size_t num_v = graph.num_vertices();
    CentralityResult result;
    result.betweenness.assign(num_v, 0.0);
    result.degree_centrality = degree_centrality(graph);

    if (num_v <= 1) {
        return result;
    }

    const auto signed_num_v = static_cast<std::ptrdiff_t>(num_v);
    constexpr bool is_weighted = !std::is_void_v<WeightType>;

#if defined(_OPENMP)
#pragma omp parallel
    {
        std::vector<double> thread_betweenness(num_v, 0.0);
        detail::BrandesWorkspace ws;
        ws.init(num_v);

        if constexpr (!is_weighted) {
            std::vector<int32_t> dist(num_v, -1);
            std::queue<VertexId> queue;

#pragma omp for schedule(dynamic, 16)
            for (std::ptrdiff_t s_idx = 0; s_idx < signed_num_v; ++s_idx) {
                const auto s = static_cast<VertexId>(s_idx);
                detail::accumulate_source_unweighted(s, graph, ws, dist, queue, thread_betweenness);
            }
        } else {
            std::vector<WeightType> dist(num_v, std::numeric_limits<WeightType>::infinity());
            d_ary_heap<4, WeightType, VertexId> heap(num_v);

#pragma omp for schedule(dynamic, 16)
            for (std::ptrdiff_t s_idx = 0; s_idx < signed_num_v; ++s_idx) {
                const auto s = static_cast<VertexId>(s_idx);
                detail::accumulate_source_weighted(s, graph, ws, dist, heap, thread_betweenness);
            }
        }

#pragma omp critical
        {
            for (std::size_t i = 0; i < num_v; ++i) {
                result.betweenness[i] += thread_betweenness[i];
            }
        }
    }
#else
    detail::BrandesWorkspace ws;
    ws.init(num_v);

    if constexpr (!is_weighted) {
        std::vector<int32_t> dist(num_v, -1);
        std::queue<VertexId> queue;

        for (VertexId s = 0; s < num_v; ++s) {
            detail::accumulate_source_unweighted(s, graph, ws, dist, queue, result.betweenness);
        }
    } else {
        std::vector<WeightType> dist(num_v, std::numeric_limits<WeightType>::infinity());
        d_ary_heap<4, WeightType, VertexId> heap(num_v);

        for (VertexId s = 0; s < num_v; ++s) {
            detail::accumulate_source_weighted(s, graph, ws, dist, heap, result.betweenness);
        }
    }
#endif

    // Halve betweenness for undirected graphs (Freeman standard)
    if (!graph.is_directed()) {
        for (std::size_t i = 0; i < num_v; ++i) {
            result.betweenness[i] *= 0.5;
        }
    }

    return result;
}

} // namespace algoat::graph

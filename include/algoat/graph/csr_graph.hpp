/**
 * @file csr_graph.hpp
 * @brief High-performance Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) graph
 * engine.
 *
 * Provides cache-conscious, zero-overhead static graph representation with contiguous
 * 64-byte aligned edge memory, O(1) degree inspection, bidirectional CSR/CSC traversal
 * for directed graphs, and zero-allocation std::span slicing.
 */

#pragma once

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::graph {

using VertexId = uint32_t;
using EdgeIndex = uint64_t;

namespace detail {

/**
 * @struct AlignedAllocator
 * @brief Standard C++20 conforming allocator providing guaranteed byte alignment for heap buffers.
 */
template <typename T, std::size_t Alignment = 64> struct AlignedAllocator {
    using value_type = T;
    static constexpr std::size_t alignment = Alignment;

    template <typename U> struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

    constexpr AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void* ptr = ::operator new(n * sizeof(T), std::align_val_t{Alignment});
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t /*n*/) noexcept {
        if (p != nullptr) {
            ::operator delete(p, std::align_val_t{Alignment});
        }
    }

    template <typename U>
    constexpr bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }

    template <typename U>
    constexpr bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

using AlignedEdgeVector = std::vector<EdgeIndex, AlignedAllocator<EdgeIndex, 64>>;
using AlignedVertexVector = std::vector<VertexId, AlignedAllocator<VertexId, 64>>;

/**
 * @brief High-throughput parallel squish and compaction pipeline for neighborhood lists.
 */
inline void compress_neighborhoods(std::size_t v_count, const std::vector<EdgeIndex>& raw_offsets,
                                   AlignedVertexVector& raw_indices, AlignedEdgeVector& out_offsets,
                                   AlignedVertexVector& out_indices) {
    bool has_duplicates = false;
    const auto signed_v_count = static_cast<std::ptrdiff_t>(v_count);

#if defined(_OPENMP)
#pragma omp parallel for reduction(| : has_duplicates) schedule(static)
#endif
    for (std::ptrdiff_t u = 0; u < signed_v_count; ++u) {
        const auto u_sz = static_cast<std::size_t>(u);
        auto* start = raw_indices.data() + raw_offsets[u_sz];
        auto* end = raw_indices.data() + raw_offsets[u_sz + 1];
        auto count = static_cast<std::size_t>(end - start);
        if (count <= 1) {
            // Already sorted and unique
        } else if (count == 2) {
            if (start[0] > start[1]) {
                std::swap(start[0], start[1]);
            } else if (start[0] == start[1]) {
                has_duplicates = true;
            }
        } else {
            std::sort(start, end);
            if (std::adjacent_find(start, end) != end) {
                has_duplicates = true;
            }
        }
    }

    if (!has_duplicates) {
        out_offsets.assign(raw_offsets.begin(), raw_offsets.end());
        out_indices = std::move(raw_indices);
    } else {
        std::vector<EdgeIndex> diff(v_count, 0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t u = 0; u < signed_v_count; ++u) {
            const auto u_sz = static_cast<std::size_t>(u);
            const auto start_idx = raw_offsets[u_sz];
            const auto end_idx = raw_offsets[u_sz + 1];
            const auto deg = end_idx - start_idx;
            if (deg <= 1) {
                diff[u_sz] = deg;
            } else if (deg == 2) {
                auto* ptr = raw_indices.data() + start_idx;
                diff[u_sz] = (ptr[0] == ptr[1]) ? 1 : 2;
            } else {
                auto* start = raw_indices.data() + start_idx;
                auto* end = raw_indices.data() + end_idx;
                auto* new_end = std::unique(start, end);
                diff[u_sz] = static_cast<EdgeIndex>(new_end - start);
            }
        }

        out_offsets.assign(v_count + 1, 0);
        for (std::size_t u = 0; u < v_count; ++u) {
            out_offsets[u + 1] = out_offsets[u] + diff[u];
        }
        out_indices.resize(out_offsets[v_count]);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t u = 0; u < signed_v_count; ++u) {
            const auto u_sz = static_cast<std::size_t>(u);
            const auto* src = raw_indices.data() + raw_offsets[u_sz];
            auto* dst = out_indices.data() + out_offsets[u_sz];
            std::copy(src, src + diff[u_sz], dst);
        }
    }
}

} // namespace detail

/**
 * @concept CSRGraphConcept
 * @brief Concept defining the public interface and requirements for static CSR graph types.
 */
template <typename G>
concept CSRGraphConcept = requires(const G& g, typename G::vertex_type u) {
    typename G::vertex_type;
    typename G::edge_index_type;
    typename G::weight_type;
    { g.num_vertices() } noexcept -> std::convertible_to<typename G::vertex_type>;
    { g.num_edges() } noexcept -> std::convertible_to<typename G::edge_index_type>;
    { g.is_directed() } noexcept -> std::same_as<bool>;
    { g.out_degree(u) } noexcept -> std::convertible_to<typename G::edge_index_type>;
    { g.out_neighbors(u) } noexcept -> std::ranges::contiguous_range;
    { g.in_neighbors(u) } noexcept -> std::ranges::contiguous_range;
    { g.transpose() } -> std::same_as<G>;
    { g.row_ptr_data() } noexcept -> std::same_as<const typename G::edge_index_type*>;
    { g.col_idx_data() } noexcept -> std::same_as<const typename G::vertex_type*>;
};

template <typename WeightType = void> class CSRGraph;

/**
 * @class CSRGraph<void>
 * @brief Primary unweighted Compressed Sparse Row graph representation.
 */
template <> class CSRGraph<void> {
public:
    using vertex_type = VertexId;
    using edge_index_type = EdgeIndex;
    using weight_type = void;

    CSRGraph() {
        row_ptr_.assign(1, 0);
    }

    /**
     * @brief Constructs an immutable CSRGraph from a raw edge list in O(V + E) time.
     *
     * @param num_vertices Number of vertices in the graph (|V|).
     * @param edges Slices of directed or undirected (u, v) pairs.
     * @param directed True if edges are directed (u -> v); false for undirected symmetric graphs.
     * @return CSRGraph instance with sorted and deduplicated neighbor lists.
     */
    static CSRGraph build_from_edges(std::size_t num_vertices,
                                     std::span<const std::pair<VertexId, VertexId>> edges,
                                     bool directed = false) {
        CSRGraph g;
        g.num_vertices_ = num_vertices;
        g.is_directed_ = directed;

        if (num_vertices == 0) {
            g.row_ptr_.assign(1, 0);
            if (directed) {
                g.col_ptr_.assign(1, 0);
            }
            return g;
        }

        const auto v_count = num_vertices;
        const auto edge_count = edges.size();

#if defined(_OPENMP)
        int num_threads = omp_get_max_threads();
        const bool use_parallel_partition =
            (num_threads > 1 && edge_count >= 100000 && v_count > 0);
#else
        const bool use_parallel_partition = false;
#endif

        if (directed) {
#if defined(_OPENMP)
            if (use_parallel_partition) {
                const std::size_t n_threads = static_cast<std::size_t>(num_threads);
                const std::size_t chunk = (edge_count + n_threads - 1) / n_threads;
                std::vector<EdgeIndex> thread_out(n_threads * v_count, 0);
                std::vector<EdgeIndex> thread_in(n_threads * v_count, 0);

#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    std::size_t start = static_cast<std::size_t>(tid) * chunk;
                    std::size_t end = std::min(start + chunk, edge_count);
                    auto* my_out = thread_out.data() + static_cast<std::size_t>(tid) * v_count;
                    auto* my_in = thread_in.data() + static_cast<std::size_t>(tid) * v_count;
                    for (std::size_t i = start; i < end; ++i) {
                        const auto& [u, v] = edges[i];
                        if (u < v_count && v < v_count) {
                            ++my_out[u];
                            ++my_in[v];
                        }
                    }
                }

                std::vector<EdgeIndex> raw_row_ptr(v_count + 1, 0);
                std::vector<EdgeIndex> raw_col_ptr(v_count + 1, 0);
                const auto signed_v_count = static_cast<std::ptrdiff_t>(v_count);

#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t u = 0; u < signed_v_count; ++u) {
                    const auto u_sz = static_cast<std::size_t>(u);
                    EdgeIndex out_sum = 0;
                    EdgeIndex in_sum = 0;
                    for (std::size_t t = 0; t < n_threads; ++t) {
                        out_sum += thread_out[t * v_count + u_sz];
                        in_sum += thread_in[t * v_count + u_sz];
                    }
                    raw_row_ptr[u_sz + 1] = out_sum;
                    raw_col_ptr[u_sz + 1] = in_sum;
                }

                raw_row_ptr[0] = 0;
                raw_col_ptr[0] = 0;
                for (std::size_t u = 0; u < v_count; ++u) {
                    raw_row_ptr[u + 1] += raw_row_ptr[u];
                    raw_col_ptr[u + 1] += raw_col_ptr[u];
                }

#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t u = 0; u < signed_v_count; ++u) {
                    const auto u_sz = static_cast<std::size_t>(u);
                    EdgeIndex out_off = raw_row_ptr[u_sz];
                    EdgeIndex in_off = raw_col_ptr[u_sz];
                    for (std::size_t t = 0; t < n_threads; ++t) {
                        const std::size_t idx = t * v_count + u_sz;
                        const EdgeIndex o_cnt = thread_out[idx];
                        const EdgeIndex i_cnt = thread_in[idx];
                        thread_out[idx] = out_off;
                        thread_in[idx] = in_off;
                        out_off += o_cnt;
                        in_off += i_cnt;
                    }
                }

                detail::AlignedVertexVector raw_col_idx(raw_row_ptr[v_count]);
                detail::AlignedVertexVector raw_row_idx(raw_col_ptr[v_count]);

#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    std::size_t start = static_cast<std::size_t>(tid) * chunk;
                    std::size_t end = std::min(start + chunk, edge_count);
                    auto* my_out = thread_out.data() + static_cast<std::size_t>(tid) * v_count;
                    auto* my_in = thread_in.data() + static_cast<std::size_t>(tid) * v_count;
                    for (std::size_t i = start; i < end; ++i) {
                        const auto& [u, v] = edges[i];
                        if (u < v_count && v < v_count) {
                            raw_col_idx[my_out[u]++] = v;
                            raw_row_idx[my_in[v]++] = u;
                        }
                    }
                }

                detail::compress_neighborhoods(v_count, raw_row_ptr, raw_col_idx, g.row_ptr_,
                                               g.col_idx_);
                detail::compress_neighborhoods(v_count, raw_col_ptr, raw_row_idx, g.col_ptr_,
                                               g.row_idx_);
                g.num_edges_ = g.col_idx_.size();
                return g;
            }
#endif

            std::vector<EdgeIndex> out_deg(v_count, 0);
            std::vector<EdgeIndex> in_deg(v_count, 0);

            for (const auto& [u, v] : edges) {
                if (u < v_count && v < v_count) {
                    ++out_deg[u];
                    ++in_deg[v];
                }
            }

            std::vector<EdgeIndex> raw_row_ptr(v_count + 1, 0);
            for (std::size_t i = 0; i < v_count; ++i) {
                raw_row_ptr[i + 1] = raw_row_ptr[i] + out_deg[i];
            }
            detail::AlignedVertexVector raw_col_idx(raw_row_ptr[v_count]);

            std::vector<EdgeIndex> raw_col_ptr(v_count + 1, 0);
            for (std::size_t i = 0; i < v_count; ++i) {
                raw_col_ptr[i + 1] = raw_col_ptr[i] + in_deg[i];
            }
            detail::AlignedVertexVector raw_row_idx(raw_col_ptr[v_count]);

            std::vector<EdgeIndex> cur_out = raw_row_ptr;
            std::vector<EdgeIndex> cur_in = raw_col_ptr;

            for (const auto& [u, v] : edges) {
                if (u < v_count && v < v_count) {
                    raw_col_idx[cur_out[u]++] = v;
                    raw_row_idx[cur_in[v]++] = u;
                }
            }

            detail::compress_neighborhoods(v_count, raw_row_ptr, raw_col_idx, g.row_ptr_,
                                           g.col_idx_);
            detail::compress_neighborhoods(v_count, raw_col_ptr, raw_row_idx, g.col_ptr_,
                                           g.row_idx_);
            g.num_edges_ = g.col_idx_.size();
        } else {
#if defined(_OPENMP)
            if (use_parallel_partition) {
                const std::size_t n_threads = static_cast<std::size_t>(num_threads);
                const std::size_t chunk = (edge_count + n_threads - 1) / n_threads;
                std::vector<EdgeIndex> thread_deg(n_threads * v_count, 0);

#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    std::size_t start = static_cast<std::size_t>(tid) * chunk;
                    std::size_t end = std::min(start + chunk, edge_count);
                    auto* my_deg = thread_deg.data() + static_cast<std::size_t>(tid) * v_count;
                    for (std::size_t i = start; i < end; ++i) {
                        const auto& [u, v] = edges[i];
                        if (u < v_count && v < v_count) {
                            ++my_deg[u];
                            if (u != v) {
                                ++my_deg[v];
                            }
                        }
                    }
                }

                std::vector<EdgeIndex> raw_row_ptr(v_count + 1, 0);
                const auto signed_v_count = static_cast<std::ptrdiff_t>(v_count);
#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t u = 0; u < signed_v_count; ++u) {
                    const auto u_sz = static_cast<std::size_t>(u);
                    EdgeIndex deg_sum = 0;
                    for (std::size_t t = 0; t < n_threads; ++t) {
                        deg_sum += thread_deg[t * v_count + u_sz];
                    }
                    raw_row_ptr[u_sz + 1] = deg_sum;
                }

                raw_row_ptr[0] = 0;
                for (std::size_t u = 0; u < v_count; ++u) {
                    raw_row_ptr[u + 1] += raw_row_ptr[u];
                }

#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t u = 0; u < signed_v_count; ++u) {
                    const auto u_sz = static_cast<std::size_t>(u);
                    EdgeIndex off = raw_row_ptr[u_sz];
                    for (std::size_t t = 0; t < n_threads; ++t) {
                        const std::size_t idx = t * v_count + u_sz;
                        const EdgeIndex cnt = thread_deg[idx];
                        thread_deg[idx] = off;
                        off += cnt;
                    }
                }

                detail::AlignedVertexVector raw_col_idx(raw_row_ptr[v_count]);

#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    std::size_t start = static_cast<std::size_t>(tid) * chunk;
                    std::size_t end = std::min(start + chunk, edge_count);
                    auto* my_deg = thread_deg.data() + static_cast<std::size_t>(tid) * v_count;
                    for (std::size_t i = start; i < end; ++i) {
                        const auto& [u, v] = edges[i];
                        if (u < v_count && v < v_count) {
                            raw_col_idx[my_deg[u]++] = v;
                            if (u != v) {
                                raw_col_idx[my_deg[v]++] = u;
                            }
                        }
                    }
                }

                detail::compress_neighborhoods(v_count, raw_row_ptr, raw_col_idx, g.row_ptr_,
                                               g.col_idx_);
                g.num_edges_ = g.col_idx_.size();
                return g;
            }
#endif

            std::vector<EdgeIndex> deg(v_count, 0);

            for (const auto& [u, v] : edges) {
                if (u < v_count && v < v_count) {
                    ++deg[u];
                    if (u != v) {
                        ++deg[v];
                    }
                }
            }

            std::vector<EdgeIndex> raw_row_ptr(v_count + 1, 0);
            for (std::size_t i = 0; i < v_count; ++i) {
                raw_row_ptr[i + 1] = raw_row_ptr[i] + deg[i];
            }
            detail::AlignedVertexVector raw_col_idx(raw_row_ptr[v_count]);

            std::vector<EdgeIndex> cur = raw_row_ptr;

            for (const auto& [u, v] : edges) {
                if (u < v_count && v < v_count) {
                    raw_col_idx[cur[u]++] = v;
                    if (u != v) {
                        raw_col_idx[cur[v]++] = u;
                    }
                }
            }

            detail::compress_neighborhoods(v_count, raw_row_ptr, raw_col_idx, g.row_ptr_,
                                           g.col_idx_);
            g.num_edges_ = g.col_idx_.size();
        }

        return g;
    }

    /**
     * @brief Builds a CSRGraph from a SNAP-format text stream.
     * Comments starting with '#' are ignored. Each line contains: FromNodeId ToNodeId.
     */
    static CSRGraph
    build_from_snap_stream(std::istream& is, bool directed = false,
                           std::optional<std::size_t> forced_num_vertices = std::nullopt) {
        std::vector<std::pair<VertexId, VertexId>> edges;
        VertexId max_v = 0;
        std::string line;

        while (std::getline(is, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream iss(line);
            VertexId u = 0;
            VertexId v = 0;
            if (iss >> u >> v) {
                edges.emplace_back(u, v);
                max_v = std::max({max_v, u, v});
            }
        }

        std::size_t num_v =
            forced_num_vertices.value_or(edges.empty() ? 0 : static_cast<std::size_t>(max_v + 1));
        return build_from_edges(num_v, edges, directed);
    }

    /**
     * @brief Builds a CSRGraph from a SNAP-format text string.
     */
    static CSRGraph
    build_from_snap_string(std::string_view text, bool directed = false,
                           std::optional<std::size_t> forced_num_vertices = std::nullopt) {
        std::istringstream iss{std::string(text)};
        return build_from_snap_stream(iss, directed, forced_num_vertices);
    }

    [[nodiscard]] VertexId num_vertices() const noexcept {
        return static_cast<VertexId>(num_vertices_);
    }

    [[nodiscard]] EdgeIndex num_edges() const noexcept {
        return num_edges_;
    }

    [[nodiscard]] bool is_directed() const noexcept {
        return is_directed_;
    }

    /**
     * @brief Returns the out-degree of vertex u in O(1) time.
     */
    [[nodiscard]] EdgeIndex out_degree(VertexId u) const noexcept {
        if (u >= num_vertices_ || row_ptr_.size() <= 1) {
            return 0;
        }
        return row_ptr_[u + 1] - row_ptr_[u];
    }

    /**
     * @brief Returns the in-degree of vertex u in O(1) time.
     */
    [[nodiscard]] EdgeIndex in_degree(VertexId u) const noexcept {
        if (u >= num_vertices_) {
            return 0;
        }
        if (!is_directed_) {
            return out_degree(u);
        }
        if (col_ptr_.size() <= 1) {
            return 0;
        }
        return col_ptr_[u + 1] - col_ptr_[u];
    }

    /**
     * @brief Contiguous slice of outgoing neighbors of vertex u with zero heap allocations.
     */
    [[nodiscard]] std::span<const VertexId> out_neighbors(VertexId u) const noexcept {
        if (u >= num_vertices_ || row_ptr_.size() <= 1) {
            return {};
        }
        const EdgeIndex start = row_ptr_[u];
        const EdgeIndex count = row_ptr_[u + 1] - start;
        return {col_idx_.data() + static_cast<std::size_t>(start), static_cast<std::size_t>(count)};
    }

    /**
     * @brief Contiguous slice of incoming neighbors of vertex u with zero heap allocations.
     */
    [[nodiscard]] std::span<const VertexId> in_neighbors(VertexId u) const noexcept {
        if (u >= num_vertices_) {
            return {};
        }
        if (!is_directed_) {
            return out_neighbors(u);
        }
        if (col_ptr_.size() <= 1) {
            return {};
        }
        const EdgeIndex start = col_ptr_[u];
        const EdgeIndex count = col_ptr_[u + 1] - start;
        return {row_idx_.data() + static_cast<std::size_t>(start), static_cast<std::size_t>(count)};
    }

    /**
     * @brief Binary search lookup for the global edge index of directed edge (u, v).
     */
    [[nodiscard]] std::optional<EdgeIndex> find_out_edge_index(VertexId u,
                                                               VertexId v) const noexcept {
        if (u >= num_vertices_ || row_ptr_.size() <= 1) {
            return std::nullopt;
        }
        auto neigh = out_neighbors(u);
        auto it = std::lower_bound(neigh.begin(), neigh.end(), v);
        if (it != neigh.end() && *it == v) {
            return row_ptr_[u] + static_cast<EdgeIndex>(it - neigh.begin());
        }
        return std::nullopt;
    }

    /**
     * @brief Binary search lookup for the global reverse edge index of directed edge (u, v).
     */
    [[nodiscard]] std::optional<EdgeIndex> find_in_edge_index(VertexId u,
                                                              VertexId v) const noexcept {
        if (!is_directed_) {
            return find_out_edge_index(v, u);
        }
        if (v >= num_vertices_ || col_ptr_.size() <= 1) {
            return std::nullopt;
        }
        auto in_neigh = in_neighbors(v);
        auto it = std::lower_bound(in_neigh.begin(), in_neigh.end(), u);
        if (it != in_neigh.end() && *it == u) {
            return col_ptr_[v] + static_cast<EdgeIndex>(it - in_neigh.begin());
        }
        return std::nullopt;
    }

    /**
     * @brief Returns the transposed graph G^T in O(V + E) time via dual CSR/CSC assignment.
     */
    [[nodiscard]] CSRGraph transpose() const {
        if (!is_directed_) {
            return *this;
        }
        CSRGraph gt;
        gt.num_vertices_ = num_vertices_;
        gt.num_edges_ = num_edges_;
        gt.is_directed_ = true;

        // Duality: CSR(A^T) == CSC(A) and CSC(A^T) == CSR(A)
        gt.row_ptr_ = col_ptr_;
        gt.col_idx_ = row_idx_;
        gt.col_ptr_ = row_ptr_;
        gt.row_idx_ = col_idx_;

        return gt;
    }

    /**
     * @brief Raw buffer accessors for vectorized SIMD SpMV kernels.
     */
    [[nodiscard]] const EdgeIndex* row_ptr_data() const noexcept {
        return row_ptr_.data();
    }

    [[nodiscard]] const VertexId* col_idx_data() const noexcept {
        return col_idx_.empty() ? nullptr : col_idx_.data();
    }

    [[nodiscard]] const EdgeIndex* col_ptr_data() const noexcept {
        return col_ptr_.empty() ? nullptr : col_ptr_.data();
    }

    [[nodiscard]] const VertexId* row_idx_data() const noexcept {
        return row_idx_.empty() ? nullptr : row_idx_.data();
    }

    /**
     * @brief Computes the exact heap memory footprint in bytes.
     */
    [[nodiscard]] std::size_t memory_footprint() const noexcept {
        std::size_t bytes = 0;
        if (!row_ptr_.empty()) {
            bytes += row_ptr_.size() * sizeof(EdgeIndex);
        }
        if (!col_idx_.empty()) {
            bytes += col_idx_.size() * sizeof(VertexId);
        }
        if (!col_ptr_.empty()) {
            bytes += col_ptr_.size() * sizeof(EdgeIndex);
        }
        if (!row_idx_.empty()) {
            bytes += row_idx_.size() * sizeof(VertexId);
        }
        return bytes;
    }

private:
    std::size_t num_vertices_{0};
    std::size_t num_edges_{0};
    bool is_directed_{false};

    // 64-byte cache line aligned storage
    alignas(64) detail::AlignedEdgeVector row_ptr_;   // Size: num_vertices_ + 1
    alignas(64) detail::AlignedVertexVector col_idx_; // Size: num_edges_
    alignas(64) detail::AlignedEdgeVector col_ptr_;   // CSC index for directed graphs
    alignas(64) detail::AlignedVertexVector row_idx_; // CSC index for directed graphs
};

/**
 * @class CSRGraph<WeightType>
 * @brief Weighted Compressed Sparse Row graph representation.
 */
template <typename WeightType> class CSRGraph {
public:
    using vertex_type = VertexId;
    using edge_index_type = EdgeIndex;
    using weight_type = WeightType;

    CSRGraph() = default;

    static CSRGraph build_from_edges(std::size_t num_vertices,
                                     std::span<const std::pair<VertexId, VertexId>> edges,
                                     bool directed = false) {
        CSRGraph g;
        g.base_ = CSRGraph<void>::build_from_edges(num_vertices, edges, directed);
        if constexpr (!std::is_void_v<WeightType>) {
            g.values_.resize(g.base_.num_edges(), WeightType{});
            if (directed) {
                g.in_values_.resize(g.base_.num_edges(), WeightType{});
            }
        }
        return g;
    }

    static CSRGraph
    build_from_snap_stream(std::istream& is, bool directed = false,
                           std::optional<std::size_t> forced_num_vertices = std::nullopt) {
        CSRGraph g;
        g.base_ = CSRGraph<void>::build_from_snap_stream(is, directed, forced_num_vertices);
        if constexpr (!std::is_void_v<WeightType>) {
            g.values_.resize(g.base_.num_edges(), WeightType{});
            if (directed) {
                g.in_values_.resize(g.base_.num_edges(), WeightType{});
            }
        }
        return g;
    }

    static CSRGraph
    build_from_snap_string(std::string_view text, bool directed = false,
                           std::optional<std::size_t> forced_num_vertices = std::nullopt) {
        std::istringstream iss{std::string(text)};
        return build_from_snap_stream(iss, directed, forced_num_vertices);
    }

    static CSRGraph
    build_from_weighted_edges(std::size_t num_vertices,
                              std::span<const std::tuple<VertexId, VertexId, WeightType>> edges,
                              bool directed = false) {
        std::vector<std::pair<VertexId, VertexId>> unweighted_edges;
        unweighted_edges.reserve(edges.size());
        for (const auto& [u, v, w] : edges) {
            unweighted_edges.emplace_back(u, v);
        }

        CSRGraph g;
        g.base_ = CSRGraph<void>::build_from_edges(num_vertices, unweighted_edges, directed);
        g.values_.resize(g.base_.num_edges(), WeightType{});
        if (directed) {
            g.in_values_.resize(g.base_.num_edges(), WeightType{});
        }

        // Map weights using the base graph's index lookups
        for (const auto& [u, v, w] : edges) {
            if (auto idx = g.base_.find_out_edge_index(u, v)) {
                g.values_[*idx] = w;
            }
            if (directed) {
                if (auto in_idx = g.base_.find_in_edge_index(u, v)) {
                    g.in_values_[*in_idx] = w;
                }
            } else if (u != v) {
                if (auto rev_idx = g.base_.find_out_edge_index(v, u)) {
                    g.values_[*rev_idx] = w;
                }
            }
        }

        return g;
    }

    [[nodiscard]] VertexId num_vertices() const noexcept {
        return base_.num_vertices();
    }
    [[nodiscard]] EdgeIndex num_edges() const noexcept {
        return base_.num_edges();
    }
    [[nodiscard]] bool is_directed() const noexcept {
        return base_.is_directed();
    }
    [[nodiscard]] EdgeIndex out_degree(VertexId u) const noexcept {
        return base_.out_degree(u);
    }
    [[nodiscard]] EdgeIndex in_degree(VertexId u) const noexcept {
        return base_.in_degree(u);
    }
    [[nodiscard]] std::span<const VertexId> out_neighbors(VertexId u) const noexcept {
        return base_.out_neighbors(u);
    }
    [[nodiscard]] std::span<const VertexId> in_neighbors(VertexId u) const noexcept {
        return base_.in_neighbors(u);
    }

    [[nodiscard]] std::span<const WeightType> out_weights(VertexId u) const noexcept {
        if (u >= num_vertices() || base_.row_ptr_data() == nullptr) {
            return {};
        }
        const EdgeIndex start = base_.row_ptr_data()[u];
        const EdgeIndex count = base_.row_ptr_data()[u + 1] - start;
        return {values_.data() + static_cast<std::size_t>(start), static_cast<std::size_t>(count)};
    }

    [[nodiscard]] std::span<const WeightType> in_weights(VertexId u) const noexcept {
        if (u >= num_vertices()) {
            return {};
        }
        if (!is_directed()) {
            return out_weights(u);
        }
        if (base_.col_ptr_data() == nullptr) {
            return {};
        }
        const EdgeIndex start = base_.col_ptr_data()[u];
        const EdgeIndex count = base_.col_ptr_data()[u + 1] - start;
        return {in_values_.data() + static_cast<std::size_t>(start),
                static_cast<std::size_t>(count)};
    }

    [[nodiscard]] CSRGraph transpose() const {
        CSRGraph gt;
        gt.base_ = base_.transpose();
        if (is_directed()) {
            gt.values_ = in_values_;
            gt.in_values_ = values_;
        } else {
            gt.values_ = values_;
        }
        return gt;
    }

    [[nodiscard]] const EdgeIndex* row_ptr_data() const noexcept {
        return base_.row_ptr_data();
    }
    [[nodiscard]] const VertexId* col_idx_data() const noexcept {
        return base_.col_idx_data();
    }
    [[nodiscard]] const EdgeIndex* col_ptr_data() const noexcept {
        return base_.col_ptr_data();
    }
    [[nodiscard]] const VertexId* row_idx_data() const noexcept {
        return base_.row_idx_data();
    }
    [[nodiscard]] const WeightType* values_data() const noexcept {
        return values_.data();
    }

    [[nodiscard]] std::size_t memory_footprint() const noexcept {
        std::size_t bytes = base_.memory_footprint();
        bytes += values_.size() * sizeof(WeightType);
        bytes += in_values_.size() * sizeof(WeightType);
        return bytes;
    }

private:
    CSRGraph<void> base_;
    alignas(64) std::vector<WeightType, detail::AlignedAllocator<WeightType, 64>> values_;
    alignas(64) std::vector<WeightType, detail::AlignedAllocator<WeightType, 64>> in_values_;
};

static_assert(CSRGraphConcept<CSRGraph<void>>, "CSRGraph<void> must satisfy CSRGraphConcept");
static_assert(CSRGraphConcept<CSRGraph<float>>, "CSRGraph<float> must satisfy CSRGraphConcept");

} // namespace algoat::graph

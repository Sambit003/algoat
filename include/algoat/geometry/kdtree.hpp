/**
 * @file kdtree.hpp
 * @brief Cache-Conscious Flat Eytzinger K-d Tree with Fast k-NN, Radius, and Box Search.
 */

#pragma once

#include "algoat/numerics/morton.hpp"
#include "algoat/searching/detail/prefetch.hpp"

#if !defined(ALGOAT_SIMD_PROFILER_SCALAR_ONLY) && !defined(ALGOAT_KDTREE_SCALAR_ONLY)
#include <hwy/highway.h>
#define ALGOAT_KDTREE_HAS_HWY 1
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ranges>
#include <span>
#include <vector>

namespace algoat::geometry {

#if defined(ALGOAT_KDTREE_HAS_HWY)
namespace simd {
namespace hn = hwy::HWY_NAMESPACE;

/**
 * @brief Highway vectorized Euclidean distance squared across 3D coordinates.
 */
template <class D, class V>
HWY_INLINE V EuclideanDistSq(D /*d*/, V q_x, V q_y, V q_z, V p_x, V p_y, V p_z) noexcept {
    const auto dx = hn::Sub(q_x, p_x);
    const auto dy = hn::Sub(q_y, p_y);
    const auto dz = hn::Sub(q_z, p_z);
    return hn::MulAdd(dx, dx, hn::MulAdd(dy, dy, hn::Mul(dz, dz)));
}

/**
 * @brief Highway vectorized Manhattan distance across 3D coordinates.
 */
template <class D, class V>
HWY_INLINE V ManhattanDist(D /*d*/, V q_x, V q_y, V q_z, V p_x, V p_y, V p_z) noexcept {
    const auto dx = hn::Abs(hn::Sub(q_x, p_x));
    const auto dy = hn::Abs(hn::Sub(q_y, p_y));
    const auto dz = hn::Abs(hn::Sub(q_z, p_z));
    return hn::Add(dx, hn::Add(dy, dz));
}

/**
 * @brief Highway vectorized Chebyshev distance across 3D coordinates.
 */
template <class D, class V>
HWY_INLINE V ChebyshevDist(D /*d*/, V q_x, V q_y, V q_z, V p_x, V p_y, V p_z) noexcept {
    const auto dx = hn::Abs(hn::Sub(q_x, p_x));
    const auto dy = hn::Abs(hn::Sub(q_y, p_y));
    const auto dz = hn::Abs(hn::Sub(q_z, p_z));
    return hn::Max(dx, hn::Max(dy, dz));
}

} // namespace simd
#endif

/**
 * @brief Concept constraining spatial point types with index-based coordinate access.
 *
 * @tparam T Point type.
 * @tparam Dim Optional dimensionality constraint (default 1).
 */
template <typename T, std::size_t Dim = 1>
concept SpatialPoint = requires(T p, std::size_t i) {
    { p[i] } -> std::convertible_to<double>;
} && (Dim > 0);

/**
 * @brief Distance metric enumeration for KDTree spatial queries.
 */
enum class MetricType : uint8_t { EuclideanL2, ManhattanL1, ChebyshevLInf, MinkowskiLp };

/**
 * @class KDTree
 * @brief Cache-conscious flat Eytzinger array indexed K-d tree engine.
 *
 * Stores coordinates in contiguous Struct-of-Arrays (SoA) layout aligned to 64 bytes.
 * Tree structure is an implicit complete binary tree (Eytzinger indexing: root at 1,
 * left child at 2i, right child at 2i+1) requiring zero pointer dereferences during queries.
 * Morton Z-order pre-sorting groups spatial coordinates prior to construction.
 *
 * @tparam PointType Type conforming to SpatialPoint concept.
 * @tparam K Dimensionality of the point space (default 3).
 */
template <SpatialPoint PointType, std::size_t K = 3> class KDTree {
    static_assert(K > 0, "KDTree dimensionality K must be strictly greater than 0");
    static_assert(SpatialPoint<PointType, K>, "PointType must satisfy SpatialPoint<PointType, K>");

public:
    using coordinate_type = double;
    static constexpr std::size_t dimensions = K;

    /**
     * @brief Result record for nearest neighbor queries.
     */
    struct NeighborResult {
        std::size_t index{0};
        double distance{0.0};

        [[nodiscard]] constexpr bool operator<(const NeighborResult& other) const noexcept {
            if (distance != other.distance) {
                return distance < other.distance;
            }
            return index < other.index;
        }

        [[nodiscard]] constexpr bool operator>(const NeighborResult& other) const noexcept {
            if (distance != other.distance) {
                return distance > other.distance;
            }
            return index > other.index;
        }

        [[nodiscard]] constexpr bool operator==(const NeighborResult& other) const noexcept {
            return index == other.index && distance == other.distance;
        }
    };

    /**
     * @brief Default constructor for an empty KD-tree.
     */
    KDTree() = default;

    /**
     * @brief Zero-copy construction from a contiguous range of points.
     * @tparam R Contiguous range type.
     * @param points Input range of PointType elements.
     */
    template <std::ranges::contiguous_range R>
        requires std::same_as<std::ranges::range_value_t<R>, PointType>
    explicit KDTree(R&& points) {
        build(std::span<const PointType>(std::ranges::data(points), std::ranges::size(points)));
    }

    /**
     * @brief Rebuilds balanced tree using Morton Z-order pre-sorting and median partitioning.
     * @param points Contiguous span of spatial points.
     */
    void build(std::span<const PointType> points) {
        size_ = points.size();
        if (size_ == 0) {
            clear();
            return;
        }

        for (std::size_t d = 0; d < K; ++d) {
            coordinates_soa_[d].resize(size_ + 1);
        }
        point_indices_.resize(size_ + 1);

        struct PointRef {
            const PointType* pt;
            std::size_t original_index;
            uint64_t morton_key;
        };

        std::vector<PointRef> item_refs(size_);
        for (std::size_t i = 0; i < size_; ++i) {
            item_refs[i] = {&points[i], i, compute_morton_key(points[i])};
        }

        // Morton Z-order pre-sorting for 2D and 3D datasets
        if constexpr (K == 2 || K == 3) {
            std::sort(item_refs.begin(), item_refs.end(),
                      [](const PointRef& a, const PointRef& b) noexcept {
                          return a.morton_key < b.morton_key;
                      });
        }

        build_recursive(1, std::span<PointRef>(item_refs), 0);
    }

    /**
     * @brief K-Nearest Neighbors search with early branch pruning and bounded priority queue.
     * @param query Target spatial query point.
     * @param k_neighbors Number of nearest neighbors to retrieve.
     * @param metric Distance metric (default EuclideanL2).
     * @return Sorted vector of NeighborResult elements (ascending distance).
     */
    [[nodiscard]] std::vector<NeighborResult>
    knn_search(const PointType& query, std::size_t k_neighbors,
               MetricType metric = MetricType::EuclideanL2) const {
        return knn_search_impl(query, k_neighbors, metric, 3.0);
    }

    /**
     * @brief K-Nearest Neighbors search with explicit Minkowski exponent parameter.
     */
    [[nodiscard]] std::vector<NeighborResult> knn_search(const PointType& query,
                                                         std::size_t k_neighbors, MetricType metric,
                                                         double p_minkowski) const {
        return knn_search_impl(query, k_neighbors, metric, p_minkowski);
    }

    /**
     * @brief Fast radius query: returns all point indices within distance radius.
     * @param query Target spatial query point.
     * @param radius Maximum query distance.
     * @param metric Distance metric (default EuclideanL2).
     * @return Sorted vector of original point indices within radius.
     */
    [[nodiscard]] std::vector<std::size_t>
    radius_search(const PointType& query, double radius,
                  MetricType metric = MetricType::EuclideanL2) const {
        return radius_search_impl(query, radius, metric, 3.0);
    }

    /**
     * @brief Fast radius query with explicit Minkowski exponent parameter.
     */
    [[nodiscard]] std::vector<std::size_t> radius_search(const PointType& query, double radius,
                                                         MetricType metric,
                                                         double p_minkowski) const {
        return radius_search_impl(query, radius, metric, p_minkowski);
    }

    /**
     * @brief Range box query: axis-aligned bounding box (AABB) intersection.
     * @param min_corner Lower-bound coordinates of the bounding box.
     * @param max_corner Upper-bound coordinates of the bounding box.
     * @return Sorted vector of original point indices strictly inside the box.
     */
    [[nodiscard]] std::vector<std::size_t> box_search(const PointType& min_corner,
                                                      const PointType& max_corner) const {
        if (size_ == 0) {
            return {};
        }

        std::array<double, K> min_c;
        std::array<double, K> max_c;
        for (std::size_t d = 0; d < K; ++d) {
            min_c[d] = static_cast<double>(min_corner[d]);
            max_c[d] = static_cast<double>(max_corner[d]);
            if (min_c[d] > max_c[d]) {
                return {};
            }
        }

        std::vector<std::size_t> results;

        std::array<std::size_t, 128> stack;
        int stack_ptr = 0;
        stack[stack_ptr++] = 1;

        while (stack_ptr > 0) {
            const std::size_t curr = stack[--stack_ptr];
            if (curr > size_) {
                continue;
            }

            const std::size_t d = split_dimension(curr);

            bool inside = true;
#pragma unroll
            for (std::size_t dim = 0; dim < K; ++dim) {
                const double coord = coordinates_soa_[dim][curr];
                if (coord < min_c[dim] || coord > max_c[dim]) {
                    inside = false;
                    break;
                }
            }
            if (inside) {
                results.push_back(point_indices_[curr]);
            }

            const double split_val = coordinates_soa_[d][curr];
            const std::size_t left_child = curr << 1;
            const std::size_t right_child = (curr << 1) + 1;

            if (min_c[d] <= split_val && left_child <= size_) {
                stack[stack_ptr++] = left_child;
            }
            if (max_c[d] >= split_val && right_child <= size_) {
                stack[stack_ptr++] = right_child;
            }
        }

        std::sort(results.begin(), results.end());
        return results;
    }

    /**
     * @brief Returns total number of indexed points.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    /**
     * @brief Checks if the tree contains no points.
     */
    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    /**
     * @brief Returns the total allocated memory footprint in bytes.
     */
    [[nodiscard]] std::size_t memory_usage_bytes() const noexcept {
        std::size_t total = 0;
        for (std::size_t d = 0; d < K; ++d) {
            total += coordinates_soa_[d].capacity() * sizeof(double);
        }
        total += point_indices_.capacity() * sizeof(std::size_t);
        return total;
    }

    /**
     * @brief Clears all indexed points and releases allocated buffers.
     */
    void clear() noexcept {
        for (std::size_t d = 0; d < K; ++d) {
            coordinates_soa_[d].clear();
        }
        point_indices_.clear();
        size_ = 0;
    }

private:
    alignas(64) std::vector<double> coordinates_soa_[K];
    std::vector<std::size_t> point_indices_;
    std::size_t size_{0};

    [[nodiscard]] static constexpr unsigned int node_depth(std::size_t node_idx) noexcept {
        return 63u - static_cast<unsigned int>(std::countl_zero(node_idx));
    }

    [[nodiscard]] static constexpr std::size_t split_dimension(std::size_t node_idx) noexcept {
        return static_cast<std::size_t>(node_depth(node_idx)) % K;
    }

    static constexpr std::size_t compute_left_subtree_size(std::size_t total_nodes) noexcept {
        if (total_nodes <= 1) {
            return 0;
        }
        const unsigned int tree_height =
            63u - static_cast<unsigned int>(std::countl_zero(total_nodes));
        const std::size_t level_capacity = std::size_t{1} << tree_height;
        const std::size_t bottom_nodes = total_nodes - (level_capacity - 1);
        const std::size_t half_capacity = level_capacity >> 1;
        return (half_capacity - 1) + std::min(bottom_nodes, half_capacity);
    }

    static uint64_t compute_morton_key(const PointType& pt) noexcept {
        if constexpr (K == 2) {
            return algoat::numerics::float_to_morton(static_cast<float>(pt[0]),
                                                     static_cast<float>(pt[1]));
        } else if constexpr (K == 3) {
            auto float_to_ordered_u32 = [](float f) noexcept -> uint32_t {
                uint32_t u;
                std::memcpy(&u, &f, sizeof(u));
                uint32_t mask = (static_cast<int32_t>(u) >> 31) | 0x80000000u;
                return (u ^ mask) >> 11;
            };
            auto split_by_2 = [](uint32_t a) noexcept -> uint64_t {
                uint64_t x = a & 0x1FFFFFULL;
                x = (x | (x << 32)) & 0x001F00000000FFFFULL;
                x = (x | (x << 16)) & 0x001F0000FF0000FFULL;
                x = (x | (x << 8)) & 0x010F00F00F00F00FULL;
                x = (x | (x << 4)) & 0x10C30C30C30C30C3ULL;
                x = (x | (x << 2)) & 0x1249249249249249ULL;
                return x;
            };
            const uint32_t x = float_to_ordered_u32(static_cast<float>(pt[0]));
            const uint32_t y = float_to_ordered_u32(static_cast<float>(pt[1]));
            const uint32_t z = float_to_ordered_u32(static_cast<float>(pt[2]));
            return split_by_2(x) | (split_by_2(y) << 1) | (split_by_2(z) << 2);
        } else {
            return 0;
        }
    }

    template <typename Ref>
    void build_recursive(std::size_t node_idx, std::span<Ref> items, std::size_t d) {
        if (items.empty() || node_idx > size_) {
            return;
        }

        const std::size_t total_nodes = items.size();
        const std::size_t left_size = compute_left_subtree_size(total_nodes);

        std::nth_element(items.begin(), items.begin() + left_size, items.end(),
                         [d](const Ref& a, const Ref& b) noexcept {
                             return static_cast<double>((*a.pt)[d]) <
                                    static_cast<double>((*b.pt)[d]);
                         });

        const auto& median_item = items[left_size];
        for (std::size_t dim = 0; dim < K; ++dim) {
            coordinates_soa_[dim][node_idx] = static_cast<double>((*median_item.pt)[dim]);
        }
        point_indices_[node_idx] = median_item.original_index;

        const std::size_t next_d = (d + 1 == K) ? 0 : d + 1;
        if (left_size > 0) {
            build_recursive(node_idx << 1, items.subspan(0, left_size), next_d);
        }
        if (total_nodes > left_size + 1) {
            build_recursive((node_idx << 1) + 1, items.subspan(left_size + 1), next_d);
        }
    }

    [[nodiscard]] std::vector<NeighborResult> knn_search_impl(const PointType& query,
                                                              std::size_t k_neighbors,
                                                              MetricType metric,
                                                              double p_minkowski) const {
        if (size_ == 0 || k_neighbors == 0) {
            return {};
        }

        std::array<double, K> q_coords;
        for (std::size_t d = 0; d < K; ++d) {
            q_coords[d] = static_cast<double>(query[d]);
        }

        const std::size_t target_count = std::min(k_neighbors, size_);
        std::vector<NeighborResult> max_heap;
        max_heap.reserve(target_count);

        auto heap_comp = [](const NeighborResult& a, const NeighborResult& b) noexcept {
            if (a.distance != b.distance) {
                return a.distance < b.distance;
            }
            return a.index < b.index;
        };

        struct StackFrame {
            std::size_t node;
            double plane_dist;
        };

        std::array<StackFrame, 128> stack;
        int stack_ptr = 0;
        stack[stack_ptr++] = {1, 0.0};

        double worst_dist = std::numeric_limits<double>::infinity();

        while (stack_ptr > 0) {
            const auto [curr, node_plane_dist] = stack[--stack_ptr];
            if (curr > size_) {
                continue;
            }

            if (max_heap.size() == target_count && node_plane_dist >= worst_dist) {
                continue;
            }

            const std::size_t d = split_dimension(curr);

            const std::size_t next_d = (d + 1 == K) ? 0 : d + 1;
            const std::size_t left_child = curr << 1;
            if (left_child <= size_) {
                algoat::searching::detail::prefetch_read(&coordinates_soa_[next_d][left_child]);
            }

            const double cur_dist = compute_point_dist(q_coords, curr, metric, p_minkowski);
            const NeighborResult candidate{point_indices_[curr], cur_dist};

            if (max_heap.size() < target_count) {
                max_heap.push_back(candidate);
                std::push_heap(max_heap.begin(), max_heap.end(), heap_comp);
                if (max_heap.size() == target_count) {
                    worst_dist = max_heap.front().distance;
                }
            } else if (heap_comp(candidate, max_heap.front())) {
                std::pop_heap(max_heap.begin(), max_heap.end(), heap_comp);
                max_heap.back() = candidate;
                std::push_heap(max_heap.begin(), max_heap.end(), heap_comp);
                worst_dist = max_heap.front().distance;
            }

            const double split_val = coordinates_soa_[d][curr];
            const double diff = q_coords[d] - split_val;

            const std::size_t go_right = static_cast<std::size_t>(diff >= 0.0);
            const std::size_t near_child = (curr << 1) + go_right;
            const std::size_t far_child = (curr << 1) + (1 ^ go_right);

            const double plane_dist = compute_plane_dist(diff, metric, p_minkowski);

            if (far_child <= size_ && (max_heap.size() < target_count || plane_dist < worst_dist)) {
                stack[stack_ptr++] = {far_child, plane_dist};
            }

            if (near_child <= size_) {
                stack[stack_ptr++] = {near_child, node_plane_dist};
            }
        }

        finalize_distances(max_heap, metric, p_minkowski);
        std::sort(max_heap.begin(), max_heap.end());
        return max_heap;
    }

    [[nodiscard]] std::vector<std::size_t> radius_search_impl(const PointType& query, double radius,
                                                              MetricType metric,
                                                              double p_minkowski) const {
        if (size_ == 0 || radius < 0.0) {
            return {};
        }

        std::array<double, K> q_coords;
        for (std::size_t d = 0; d < K; ++d) {
            q_coords[d] = static_cast<double>(query[d]);
        }

        double threshold = 0.0;
        if (metric == MetricType::EuclideanL2) {
            threshold = radius * radius;
        } else if (metric == MetricType::ManhattanL1 || metric == MetricType::ChebyshevLInf) {
            threshold = radius;
        } else {
            threshold = std::pow(radius, p_minkowski);
        }

        std::vector<std::size_t> results;

        std::array<std::size_t, 128> stack;
        int stack_ptr = 0;
        stack[stack_ptr++] = 1;

        while (stack_ptr > 0) {
            const std::size_t curr = stack[--stack_ptr];
            if (curr > size_) {
                continue;
            }

            const std::size_t d = split_dimension(curr);

            const double cur_dist = compute_point_dist(q_coords, curr, metric, p_minkowski);
            if (cur_dist <= threshold) {
                results.push_back(point_indices_[curr]);
            }

            const double split_val = coordinates_soa_[d][curr];
            const double diff = q_coords[d] - split_val;

            const std::size_t go_right = static_cast<std::size_t>(diff >= 0.0);
            const std::size_t near_child = (curr << 1) + go_right;
            const std::size_t far_child = (curr << 1) + (1 ^ go_right);

            const double plane_dist = compute_plane_dist(diff, metric, p_minkowski);

            if (far_child <= size_ && plane_dist <= threshold) {
                stack[stack_ptr++] = far_child;
            }

            if (near_child <= size_) {
                stack[stack_ptr++] = near_child;
            }
        }

        std::sort(results.begin(), results.end());
        return results;
    }

    [[nodiscard]] inline double compute_point_dist(const std::array<double, K>& q, std::size_t idx,
                                                   MetricType metric, double p) const noexcept {
#if defined(ALGOAT_KDTREE_HAS_HWY)
        if constexpr (K == 3) {
            namespace hn = hwy::HWY_NAMESPACE;
            const hn::ScalableTag<double> d;
            if (metric == MetricType::EuclideanL2) {
                const auto q0 = hn::Set(d, q[0]);
                const auto q1 = hn::Set(d, q[1]);
                const auto q2 = hn::Set(d, q[2]);
                const auto p0 = hn::Set(d, coordinates_soa_[0][idx]);
                const auto p1 = hn::Set(d, coordinates_soa_[1][idx]);
                const auto p2 = hn::Set(d, coordinates_soa_[2][idx]);
                const auto dist_v = simd::EuclideanDistSq(d, q0, q1, q2, p0, p1, p2);
                return hn::GetLane(dist_v);
            }
            if (metric == MetricType::ManhattanL1) {
                const auto q0 = hn::Set(d, q[0]);
                const auto q1 = hn::Set(d, q[1]);
                const auto q2 = hn::Set(d, q[2]);
                const auto p0 = hn::Set(d, coordinates_soa_[0][idx]);
                const auto p1 = hn::Set(d, coordinates_soa_[1][idx]);
                const auto p2 = hn::Set(d, coordinates_soa_[2][idx]);
                const auto dist_v = simd::ManhattanDist(d, q0, q1, q2, p0, p1, p2);
                return hn::GetLane(dist_v);
            }
            if (metric == MetricType::ChebyshevLInf) {
                const auto q0 = hn::Set(d, q[0]);
                const auto q1 = hn::Set(d, q[1]);
                const auto q2 = hn::Set(d, q[2]);
                const auto p0 = hn::Set(d, coordinates_soa_[0][idx]);
                const auto p1 = hn::Set(d, coordinates_soa_[1][idx]);
                const auto p2 = hn::Set(d, coordinates_soa_[2][idx]);
                const auto dist_v = simd::ChebyshevDist(d, q0, q1, q2, p0, p1, p2);
                return hn::GetLane(dist_v);
            }
        }
#endif

        if (metric == MetricType::EuclideanL2) {
            double sum = 0.0;
#pragma unroll
            for (std::size_t d = 0; d < K; ++d) {
                const double diff = q[d] - coordinates_soa_[d][idx];
                sum += diff * diff;
            }
            return sum;
        }
        if (metric == MetricType::ManhattanL1) {
            double sum = 0.0;
#pragma unroll
            for (std::size_t d = 0; d < K; ++d) {
                sum += std::abs(q[d] - coordinates_soa_[d][idx]);
            }
            return sum;
        }
        if (metric == MetricType::ChebyshevLInf) {
            double m = 0.0;
#pragma unroll
            for (std::size_t d = 0; d < K; ++d) {
                m = std::max(m, std::abs(q[d] - coordinates_soa_[d][idx]));
            }
            return m;
        }
        // MinkowskiLp
        double sum = 0.0;
        for (std::size_t d = 0; d < K; ++d) {
            sum += std::pow(std::abs(q[d] - coordinates_soa_[d][idx]), p);
        }
        return sum;
    }

    [[nodiscard]] inline double compute_plane_dist(double diff, MetricType metric,
                                                   double p) const noexcept {
        if (metric == MetricType::EuclideanL2) {
            return diff * diff;
        }
        if (metric == MetricType::ManhattanL1 || metric == MetricType::ChebyshevLInf) {
            return std::abs(diff);
        }
        return std::pow(std::abs(diff), p);
    }

    void finalize_distances(std::span<NeighborResult> res, MetricType metric,
                            double p) const noexcept {
        if (metric == MetricType::EuclideanL2) {
            for (auto& item : res) {
                item.distance = std::sqrt(item.distance);
            }
        } else if (metric == MetricType::MinkowskiLp) {
            const double inv_p = 1.0 / p;
            for (auto& item : res) {
                item.distance = std::pow(item.distance, inv_p);
            }
        }
    }
};

} // namespace algoat::geometry

/**
 * @file lbvh.hpp
 * @brief Linear Bounding Volume Hierarchy (LBVH) via Morton-Coded Radix Sort & SIMD Traversal.
 */

#pragma once

#include "algoat/numerics/morton.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__BMI2__)
#include <immintrin.h>
#endif
#endif

namespace algoat::geometry {

/**
 * @struct AABB
 * @brief 32-byte aligned 3D Axis-Aligned Bounding Box.
 */
struct alignas(32) AABB {
    float min_x{std::numeric_limits<float>::infinity()};
    float min_y{std::numeric_limits<float>::infinity()};
    float min_z{std::numeric_limits<float>::infinity()};
    float _pad0{0.0f};
    float max_x{-std::numeric_limits<float>::infinity()};
    float max_y{-std::numeric_limits<float>::infinity()};
    float max_z{-std::numeric_limits<float>::infinity()};
    float _pad1{0.0f};

    [[nodiscard]] constexpr bool intersects(const AABB& other) const noexcept {
        return (min_x <= other.max_x && max_x >= other.min_x) &&
               (min_y <= other.max_y && max_y >= other.min_y) &&
               (min_z <= other.max_z && max_z >= other.min_z);
    }

    void expand(const AABB& other) noexcept {
        min_x = std::min(min_x, other.min_x);
        min_y = std::min(min_y, other.min_y);
        min_z = std::min(min_z, other.min_z);
        max_x = std::max(max_x, other.max_x);
        max_y = std::max(max_y, other.max_y);
        max_z = std::max(max_z, other.max_z);
    }

    [[nodiscard]] constexpr std::array<float, 3> center() const noexcept {
        return {0.5f * (min_x + max_x), 0.5f * (min_y + max_y), 0.5f * (min_z + max_z)};
    }

    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return min_x <= max_x && min_y <= max_y && min_z <= max_z;
    }
};
static_assert(sizeof(AABB) == 32, "AABB must be exactly 32 bytes");

/**
 * @struct Ray
 * @brief 16-byte aligned 3D Ray with precomputed inverse directions.
 */
struct alignas(16) Ray {
    float origin_x{0.0f}, origin_y{0.0f}, origin_z{0.0f};
    float dir_x{0.0f}, dir_y{0.0f}, dir_z{0.0f};
    float inv_dir_x{0.0f}, inv_dir_y{0.0f}, inv_dir_z{0.0f}; // Precomputed 1.0f / dir
    float t_min{0.0f};
    float t_max{std::numeric_limits<float>::infinity()};

    constexpr Ray() noexcept = default;

    constexpr Ray(float ox, float oy, float oz, float dx, float dy, float dz, float tmin = 0.0f,
                  float tmax = std::numeric_limits<float>::infinity()) noexcept
        : origin_x(ox), origin_y(oy), origin_z(oz), dir_x(dx), dir_y(dy), dir_z(dz),
          inv_dir_x(1.0f / dx), inv_dir_y(1.0f / dy), inv_dir_z(1.0f / dz), t_min(tmin),
          t_max(tmax) {}
};

/**
 * @struct BVHNode
 * @brief Flat 64-byte aligned and sized BVH Node.
 */
struct alignas(64) BVHNode {
    AABB bbox;
    uint32_t left_child{0};  // If leaf: primitive index
    uint32_t right_child{0}; // If leaf: 0xFFFFFFFF
    uint32_t parent{0xFFFFFFFF};
    bool is_leaf{false};
    uint8_t _padding[19]{0};
};
static_assert(sizeof(BVHNode) == 64, "BVHNode must be strictly 64 bytes");

/**
 * @brief Evaluates ray-AABB intersection using the Kay-Kajiya slab algorithm.
 */
inline bool intersect_ray_aabb(const Ray& ray, const AABB& box, float& hit_t_enter) noexcept {
    float t1 = (box.min_x - ray.origin_x) * ray.inv_dir_x;
    float t2 = (box.max_x - ray.origin_x) * ray.inv_dir_x;
    float tmin = std::min(t1, t2);
    float tmax = std::max(t1, t2);

    float t3 = (box.min_y - ray.origin_y) * ray.inv_dir_y;
    float t4 = (box.max_y - ray.origin_y) * ray.inv_dir_y;
    tmin = std::max(tmin, std::min(t3, t4));
    tmax = std::min(tmax, std::max(t3, t4));

    float t5 = (box.min_z - ray.origin_z) * ray.inv_dir_z;
    float t6 = (box.max_z - ray.origin_z) * ray.inv_dir_z;
    tmin = std::max(tmin, std::min(t5, t6));
    tmax = std::min(tmax, std::max(t5, t6));

    tmin = std::max(tmin, ray.t_min);
    tmax = std::min(tmax, ray.t_max);

    if (tmin <= tmax && tmax >= 0.0f) {
        hit_t_enter = tmin;
        return true;
    }
    return false;
}

namespace detail {

/**
 * @struct MortonItem
 * @brief Morton key paired with original primitive index for cache-friendly sorting.
 */
struct MortonItem {
    uint64_t code{0};
    uint32_t index{0};
    uint32_t _pad{0};
};
static_assert(sizeof(MortonItem) == 16, "MortonItem must be 16 bytes for cache alignment");

/**
 * @brief Trait helper to extract an AABB from a user-supplied Primitive.
 */
template <typename P> inline constexpr AABB extract_aabb(const P& p) noexcept {
    if constexpr (std::is_same_v<P, AABB>) {
        return p;
    } else if constexpr (requires {
                             { p.bbox } -> std::convertible_to<AABB>;
                         }) {
        return static_cast<AABB>(p.bbox);
    } else if constexpr (requires {
                             { p.bounding_box() } -> std::convertible_to<AABB>;
                         }) {
        return static_cast<AABB>(p.bounding_box());
    } else if constexpr (requires {
                             { p.bbox() } -> std::convertible_to<AABB>;
                         }) {
        return static_cast<AABB>(p.bbox());
    } else if constexpr (requires {
                             { get_bounding_box(p) } -> std::convertible_to<AABB>;
                         }) {
        return static_cast<AABB>(get_bounding_box(p));
    } else {
        static_assert(sizeof(P) == 0, "Primitive type must provide an AABB");
    }
}

/**
 * @brief Common prefix length delta(i, j) with index tie-breaking for duplicate Morton codes.
 */
inline int karras_delta(int i, int j, const uint64_t* keys, int n) noexcept {
    if (j < 0 || j >= n) {
        return -1;
    }
    uint64_t ki = keys[i];
    uint64_t kj = keys[j];
    if (ki != kj) {
        return std::countl_zero(ki ^ kj);
    }
    return 64 + std::countl_zero(static_cast<uint32_t>(i ^ j));
}

/**
 * @brief High-performance 4-pass 16-bit Radix Sort across 64-bit Morton keys.
 */
inline void radix_sort_morton(std::span<MortonItem> items) {
    const std::size_t n = items.size();
    if (n <= 1) {
        return;
    }

    if (n < 64) {
        std::stable_sort(items.begin(), items.end(),
                         [](const auto& a, const auto& b) { return a.code < b.code; });
        return;
    }

    std::vector<MortonItem> temp(n);
    MortonItem* src = items.data();
    MortonItem* dst = temp.data();

    for (int shift = 0; shift < 64; shift += 16) {
        uint32_t count[65536] = {0};
        for (std::size_t i = 0; i < n; ++i) {
            uint16_t bucket = static_cast<uint16_t>((src[i].code >> shift) & 0xFFFF);
            count[bucket]++;
        }

        bool single_bucket = false;
        for (int i = 0; i < 65536; ++i) {
            if (count[i] == n) {
                single_bucket = true;
                break;
            }
        }
        if (single_bucket) {
            continue;
        }

        uint32_t offset[65536];
        offset[0] = 0;
        for (int i = 1; i < 65536; ++i) {
            offset[i] = offset[i - 1] + count[i - 1];
        }

        for (std::size_t i = 0; i < n; ++i) {
            uint16_t bucket = static_cast<uint16_t>((src[i].code >> shift) & 0xFFFF);
            dst[offset[bucket]++] = src[i];
        }

        std::swap(src, dst);
    }

    if (src != items.data()) {
        std::copy(temp.begin(), temp.end(), items.begin());
    }
}

#if defined(__AVX2__)
/**
 * @brief 4-box AVX2 SIMD Slab test evaluating 4 bounding volumes simultaneously.
 */
inline int intersect_ray_4boxes_avx2(const Ray& ray, __m256 bbox_min4_x, __m256 bbox_min4_y,
                                     __m256 bbox_min4_z, __m256 bbox_max4_x, __m256 bbox_max4_y,
                                     __m256 bbox_max4_z, float* t_near4_out) noexcept {
    __m256 ray_org_x = _mm256_set1_ps(ray.origin_x);
    __m256 ray_org_y = _mm256_set1_ps(ray.origin_y);
    __m256 ray_org_z = _mm256_set1_ps(ray.origin_z);

    __m256 ray_inv_x = _mm256_set1_ps(ray.inv_dir_x);
    __m256 ray_inv_y = _mm256_set1_ps(ray.inv_dir_y);
    __m256 ray_inv_z = _mm256_set1_ps(ray.inv_dir_z);

    __m256 t1_x = _mm256_mul_ps(_mm256_sub_ps(bbox_min4_x, ray_org_x), ray_inv_x);
    __m256 t2_x = _mm256_mul_ps(_mm256_sub_ps(bbox_max4_x, ray_org_x), ray_inv_x);
    __m256 t_near_x = _mm256_min_ps(t1_x, t2_x);
    __m256 t_far_x = _mm256_max_ps(t1_x, t2_x);

    __m256 t1_y = _mm256_mul_ps(_mm256_sub_ps(bbox_min4_y, ray_org_y), ray_inv_y);
    __m256 t2_y = _mm256_mul_ps(_mm256_sub_ps(bbox_max4_y, ray_org_y), ray_inv_y);
    __m256 t_near_y = _mm256_min_ps(t1_y, t2_y);
    __m256 t_far_y = _mm256_max_ps(t1_y, t2_y);

    __m256 t1_z = _mm256_mul_ps(_mm256_sub_ps(bbox_min4_z, ray_org_z), ray_inv_z);
    __m256 t2_z = _mm256_mul_ps(_mm256_sub_ps(bbox_max4_z, ray_org_z), ray_inv_z);
    __m256 t_near_z = _mm256_min_ps(t1_z, t2_z);
    __m256 t_far_z = _mm256_max_ps(t1_z, t2_z);

    __m256 t_near = _mm256_max_ps(_mm256_max_ps(t_near_x, t_near_y), t_near_z);
    __m256 t_far = _mm256_min_ps(_mm256_min_ps(t_far_x, t_far_y), t_far_z);

    t_near = _mm256_max_ps(t_near, _mm256_set1_ps(ray.t_min));
    t_far = _mm256_min_ps(t_far, _mm256_set1_ps(ray.t_max));

    __m256 hit_mask = _mm256_and_ps(_mm256_cmp_ps(t_near, t_far, _CMP_LE_OQ),
                                    _mm256_cmp_ps(t_far, _mm256_setzero_ps(), _CMP_GE_OQ));
    int mask = _mm256_movemask_ps(hit_mask);
    _mm256_storeu_ps(t_near4_out, t_near);
    return mask & 0x0F;
}

/**
 * @brief 2-AABB vectorized ray intersection test.
 */
inline void intersect_ray_2aabbs_simd(const Ray& ray, const AABB& boxA, const AABB& boxB,
                                      bool& hitA, float& t_enterA, bool& hitB,
                                      float& t_enterB) noexcept {
    __m128 ray_org_x = _mm_set1_ps(ray.origin_x);
    __m128 ray_org_y = _mm_set1_ps(ray.origin_y);
    __m128 ray_org_z = _mm_set1_ps(ray.origin_z);

    __m128 ray_inv_x = _mm_set1_ps(ray.inv_dir_x);
    __m128 ray_inv_y = _mm_set1_ps(ray.inv_dir_y);
    __m128 ray_inv_z = _mm_set1_ps(ray.inv_dir_z);

    __m128 min_x = _mm_set_ps(0.0f, 0.0f, boxB.min_x, boxA.min_x);
    __m128 max_x = _mm_set_ps(0.0f, 0.0f, boxB.max_x, boxA.max_x);
    __m128 min_y = _mm_set_ps(0.0f, 0.0f, boxB.min_y, boxA.min_y);
    __m128 max_y = _mm_set_ps(0.0f, 0.0f, boxB.max_y, boxA.max_y);
    __m128 min_z = _mm_set_ps(0.0f, 0.0f, boxB.min_z, boxA.min_z);
    __m128 max_z = _mm_set_ps(0.0f, 0.0f, boxB.max_z, boxA.max_z);

    __m128 t1_x = _mm_mul_ps(_mm_sub_ps(min_x, ray_org_x), ray_inv_x);
    __m128 t2_x = _mm_mul_ps(_mm_sub_ps(max_x, ray_org_x), ray_inv_x);
    __m128 tmin = _mm_min_ps(t1_x, t2_x);
    __m128 tmax = _mm_max_ps(t1_x, t2_x);

    __m128 t1_y = _mm_mul_ps(_mm_sub_ps(min_y, ray_org_y), ray_inv_y);
    __m128 t2_y = _mm_mul_ps(_mm_sub_ps(max_y, ray_org_y), ray_inv_y);
    tmin = _mm_max_ps(tmin, _mm_min_ps(t1_y, t2_y));
    tmax = _mm_min_ps(tmax, _mm_max_ps(t1_y, t2_y));

    __m128 t1_z = _mm_mul_ps(_mm_sub_ps(min_z, ray_org_z), ray_inv_z);
    __m128 t2_z = _mm_mul_ps(_mm_sub_ps(max_z, ray_org_z), ray_inv_z);
    tmin = _mm_max_ps(tmin, _mm_min_ps(t1_z, t2_z));
    tmax = _mm_min_ps(tmax, _mm_max_ps(t1_z, t2_z));

    tmin = _mm_max_ps(tmin, _mm_set1_ps(ray.t_min));
    tmax = _mm_min_ps(tmax, _mm_set1_ps(ray.t_max));

    __m128 mask = _mm_and_ps(_mm_cmple_ps(tmin, tmax), _mm_cmpge_ps(tmax, _mm_setzero_ps()));
    int m = _mm_movemask_ps(mask);

    alignas(16) float t_arr[4];
    _mm_store_ps(t_arr, tmin);

    hitA = (m & 1) != 0;
    t_enterA = t_arr[0];
    hitB = (m & 2) != 0;
    t_enterB = t_arr[1];
}
#endif

} // namespace detail

// Expose detail types for tests/benchmarks backwards compatibility
using detail::MortonItem;
using detail::radix_sort_morton;
using numerics::dilate_3;
using numerics::morton3d_encode;
using numerics::morton3d_encode_fallback;

/**
 * @class LBVH
 * @brief Linear Bounding Volume Hierarchy engine based on Tero Karras' 2012 algorithm.
 *
 * @tparam Primitive Geometry primitive type (AABB or custom type satisfying extract_aabb).
 */
template <typename Primitive> class LBVH {
public:
    LBVH() = default;

    /**
     * @brief Linear-time BVH construction from primitive bounding boxes.
     *
     * Decouples tree generation into:
     * 1. Centroid normalization and 3D Morton encoding (BMI2 _pdep_u64 accelerated).
     * 2. 16-bit 4-pass Radix Sort across 64-bit keys.
     * 3. Parallel Radix Tree split search in O(1) time per internal node (Karras 2012).
     * 4. Atomic bottom-up bounding box fitting.
     *
     * @param primitives Contiguous span of primitives to build into the BVH.
     */
    void build(std::span<const Primitive> primitives) {
        clear();
        primitive_count_ = primitives.size();
        if (primitive_count_ == 0) {
            return;
        }

        if (primitive_count_ == 1) {
            nodes_.resize(1);
            nodes_[0].bbox = detail::extract_aabb(primitives[0]);
            nodes_[0].left_child = 0;
            nodes_[0].right_child = 0xFFFFFFFF;
            nodes_[0].parent = 0xFFFFFFFF;
            nodes_[0].is_leaf = true;
            root_index_ = 0;
            return;
        }

        const std::size_t n = primitive_count_;

        AABB centroid_bounds;
        std::vector<AABB> primitive_boxes(n);
        for (std::size_t i = 0; i < n; ++i) {
            primitive_boxes[i] = detail::extract_aabb(primitives[i]);
            auto c = primitive_boxes[i].center();
            centroid_bounds.expand(AABB{c[0], c[1], c[2], 0.0f, c[0], c[1], c[2], 0.0f});
        }

        float extent_x = centroid_bounds.max_x - centroid_bounds.min_x;
        float extent_y = centroid_bounds.max_y - centroid_bounds.min_y;
        float extent_z = centroid_bounds.max_z - centroid_bounds.min_z;

        float inv_ex = (extent_x > 1e-9f) ? (1.0f / extent_x) : 0.0f;
        float inv_ey = (extent_y > 1e-9f) ? (1.0f / extent_y) : 0.0f;
        float inv_ez = (extent_z > 1e-9f) ? (1.0f / extent_z) : 0.0f;

        std::vector<detail::MortonItem> morton_items(n);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n > 10000)
#endif
        for (std::size_t i = 0; i < n; ++i) {
            auto c = primitive_boxes[i].center();
            float nx = (c[0] - centroid_bounds.min_x) * inv_ex;
            float ny = (c[1] - centroid_bounds.min_y) * inv_ey;
            float nz = (c[2] - centroid_bounds.min_z) * inv_ez;

            nx = std::clamp(nx, 0.0f, 1.0f);
            ny = std::clamp(ny, 0.0f, 1.0f);
            nz = std::clamp(nz, 0.0f, 1.0f);

            uint32_t qx = std::min(static_cast<uint32_t>(nx * 2097152.0f), 2097151u);
            uint32_t qy = std::min(static_cast<uint32_t>(ny * 2097152.0f), 2097151u);
            uint32_t qz = std::min(static_cast<uint32_t>(nz * 2097152.0f), 2097151u);

            morton_items[i] = detail::MortonItem{numerics::morton3d_encode(qx, qy, qz),
                                                 static_cast<uint32_t>(i), 0};
        }

        detail::radix_sort_morton(morton_items);

        morton_keys_.resize(n);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n > 10000)
#endif
        for (std::size_t i = 0; i < n; ++i) {
            morton_keys_[i] = morton_items[i].code;
        }

        nodes_.resize(2 * n - 1);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n > 10000)
#endif
        for (std::size_t i = 0; i < n; ++i) {
            uint32_t leaf_idx = static_cast<uint32_t>(n - 1 + i);
            nodes_[leaf_idx].bbox = primitive_boxes[morton_items[i].index];
            nodes_[leaf_idx].left_child = morton_items[i].index;
            nodes_[leaf_idx].right_child = 0xFFFFFFFF;
            nodes_[leaf_idx].parent = 0xFFFFFFFF;
            nodes_[leaf_idx].is_leaf = true;
        }

        const uint64_t* keys = morton_keys_.data();
        const int num_prims = static_cast<int>(n);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (num_prims > 10000)
#endif
        for (int i = 0; i < num_prims - 1; ++i) {
            // Direction: compare prefix lengths with neighbors
            int direction = (detail::karras_delta(i, i + 1, keys, num_prims) >=
                             detail::karras_delta(i, i - 1, keys, num_prims))
                                ? 1
                                : -1;
            int min_prefix = detail::karras_delta(i, i - direction, keys, num_prims);

            // Exponential doubling to find range upper bound
            int max_search_stride = 2;
            while (detail::karras_delta(i, i + max_search_stride * direction, keys, num_prims) >
                   min_prefix) {
                max_search_stride <<= 1;
            }

            // Binary search for exact range endpoint
            int search_step = 0;
            for (int t = max_search_stride >> 1; t > 0; t >>= 1) {
                if (detail::karras_delta(i, i + (search_step + t) * direction, keys, num_prims) >
                    min_prefix) {
                    search_step += t;
                }
            }

            int range_endpoint = i + search_step * direction;
            int range_first = std::min(i, range_endpoint);
            int range_last = std::max(i, range_endpoint);

            // Binary search for split point gamma dividing the node
            int node_prefix = detail::karras_delta(range_first, range_last, keys, num_prims);
            int split_index = range_first;
            int stride = range_last - range_first;
            do {
                stride = (stride + 1) >> 1;
                if (split_index + stride < range_last &&
                    detail::karras_delta(range_first, split_index + stride, keys, num_prims) >
                        node_prefix) {
                    split_index += stride;
                }
            } while (stride > 1);

            uint32_t left_child = (range_first == split_index)
                                      ? static_cast<uint32_t>(n - 1 + split_index)
                                      : static_cast<uint32_t>(split_index);
            uint32_t right_child = (range_last == split_index + 1)
                                       ? static_cast<uint32_t>(n - 1 + split_index + 1)
                                       : static_cast<uint32_t>(split_index + 1);

            nodes_[i].left_child = left_child;
            nodes_[i].right_child = right_child;
            nodes_[i].is_leaf = false;
            nodes_[left_child].parent = static_cast<uint32_t>(i);
            nodes_[right_child].parent = static_cast<uint32_t>(i);
        }

        root_index_ = 0xFFFFFFFF;
        for (std::size_t i = 0; i < n - 1; ++i) {
            if (nodes_[i].parent == 0xFFFFFFFF) {
                root_index_ = static_cast<uint32_t>(i);
                break;
            }
        }

        std::vector<std::atomic<uint32_t>> flags(n - 1);
        for (std::size_t i = 0; i < n - 1; ++i) {
            flags[i].store(0, std::memory_order_relaxed);
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n > 10000)
#endif
        for (std::size_t i = 0; i < n; ++i) {
            uint32_t curr = nodes_[n - 1 + i].parent;
            while (curr != 0xFFFFFFFF) {
                if (flags[curr].fetch_add(1, std::memory_order_acq_rel) == 0) {
                    break;
                }
                nodes_[curr].bbox = nodes_[nodes_[curr].left_child].bbox;
                nodes_[curr].bbox.expand(nodes_[nodes_[curr].right_child].bbox);
                curr = nodes_[curr].parent;
            }
        }
    }

    /**
     * @brief Fast SIMD ray intersection returning the nearest primitive hit.
     *
     * @tparam RayIntersectFn Callback type invoked for leaf primitives.
     * @param ray Ray to test.
     * @param primitive_test Callable (Ray, prim_id) -> std::optional<float>, float, or bool.
     * @return Optional nearest hit primitive index.
     */
    template <typename RayIntersectFn>
    [[nodiscard]] std::optional<uint32_t> query_ray(const Ray& ray,
                                                    RayIntersectFn&& primitive_test) const {
        if (nodes_.empty() || root_index_ == 0xFFFFFFFF) {
            return std::nullopt;
        }

        struct StackEntry {
            uint32_t node_idx;
            float t_enter;
        };

        StackEntry local_stack[64];
        std::vector<StackEntry> overflow_stack;
        int local_ptr = 0;

        float root_t_enter = 0.0f;
        if (!intersect_ray_aabb(ray, nodes_[root_index_].bbox, root_t_enter)) {
            return std::nullopt;
        }

        local_stack[local_ptr++] = {root_index_, root_t_enter};

        auto push_node = [&](uint32_t n_idx, float t_ent) {
            if (local_ptr < 64) {
                local_stack[local_ptr++] = {n_idx, t_ent};
            } else {
                overflow_stack.push_back({n_idx, t_ent});
            }
        };

        auto pop_node = [&]() -> StackEntry {
            if (!overflow_stack.empty()) {
                auto top = overflow_stack.back();
                overflow_stack.pop_back();
                return top;
            }
            return local_stack[--local_ptr];
        };

        auto stack_empty = [&]() noexcept -> bool {
            return local_ptr == 0 && overflow_stack.empty();
        };

        Ray current_ray = ray;
        std::optional<uint32_t> best_hit_prim = std::nullopt;

        while (!stack_empty()) {
            StackEntry entry = pop_node();

            if (entry.t_enter >= current_ray.t_max) {
                continue;
            }

            const auto& node = nodes_[entry.node_idx];

            if (node.is_leaf) {
                uint32_t prim_id = node.left_child;
                using RetType = std::decay_t<decltype(primitive_test(current_ray, prim_id))>;
                if constexpr (std::is_same_v<RetType, std::optional<float>>) {
                    auto hit_dist = primitive_test(current_ray, prim_id);
                    if (hit_dist.has_value() && *hit_dist < current_ray.t_max &&
                        *hit_dist >= current_ray.t_min) {
                        current_ray.t_max = *hit_dist;
                        best_hit_prim = prim_id;
                    }
                } else if constexpr (std::is_same_v<RetType, bool>) {
                    if (primitive_test(current_ray, prim_id)) {
                        best_hit_prim = prim_id;
                        current_ray.t_max = entry.t_enter;
                    }
                } else if constexpr (std::is_floating_point_v<RetType>) {
                    float hit_dist = static_cast<float>(primitive_test(current_ray, prim_id));
                    if (hit_dist < current_ray.t_max && hit_dist >= current_ray.t_min) {
                        current_ray.t_max = hit_dist;
                        best_hit_prim = prim_id;
                    }
                }
                continue;
            }

            uint32_t left_child = node.left_child;
            uint32_t right_child = node.right_child;

            bool hit_left = false;
            float t_left = 0.0f;
            bool hit_right = false;
            float t_right = 0.0f;

#if defined(__AVX2__)
            detail::intersect_ray_2aabbs_simd(current_ray, nodes_[left_child].bbox,
                                              nodes_[right_child].bbox, hit_left, t_left, hit_right,
                                              t_right);
#else
            hit_left = intersect_ray_aabb(current_ray, nodes_[left_child].bbox, t_left);
            hit_right = intersect_ray_aabb(current_ray, nodes_[right_child].bbox, t_right);
#endif

            if (hit_left && hit_right) {
                if (t_left < t_right) {
                    push_node(right_child, t_right);
                    push_node(left_child, t_left);
                } else {
                    push_node(left_child, t_left);
                    push_node(right_child, t_right);
                }
            } else if (hit_left) {
                push_node(left_child, t_left);
            } else if (hit_right) {
                push_node(right_child, t_right);
            }
        }

        return best_hit_prim;
    }

    /**
     * @brief AABB overlap query returning indices of all overlapping primitives.
     *
     * @param box Query bounding box.
     * @return Vector of matching primitive indices.
     */
    [[nodiscard]] std::vector<uint32_t> query_box(const AABB& box) const {
        if (nodes_.empty() || root_index_ == 0xFFFFFFFF) {
            return {};
        }

        if (!nodes_[root_index_].bbox.intersects(box)) {
            return {};
        }

        std::vector<uint32_t> results;
        uint32_t local_stack[64];
        std::vector<uint32_t> overflow_stack;
        int local_ptr = 0;

        local_stack[local_ptr++] = root_index_;

        auto push_node = [&](uint32_t idx) {
            if (local_ptr < 64) {
                local_stack[local_ptr++] = idx;
            } else {
                overflow_stack.push_back(idx);
            }
        };

        auto pop_node = [&]() -> uint32_t {
            if (!overflow_stack.empty()) {
                uint32_t top = overflow_stack.back();
                overflow_stack.pop_back();
                return top;
            }
            return local_stack[--local_ptr];
        };

        while (local_ptr > 0 || !overflow_stack.empty()) {
            uint32_t idx = pop_node();
            const auto& node = nodes_[idx];

            if (node.is_leaf) {
                if (node.bbox.intersects(box)) {
                    results.push_back(node.left_child);
                }
            } else {
                uint32_t left = node.left_child;
                uint32_t right = node.right_child;
                if (nodes_[right].bbox.intersects(box)) {
                    push_node(right);
                }
                if (nodes_[left].bbox.intersects(box)) {
                    push_node(left);
                }
            }
        }

        return results;
    }

    [[nodiscard]] std::size_t node_count() const noexcept {
        return nodes_.size();
    }
    [[nodiscard]] std::size_t leaf_count() const noexcept {
        return primitive_count_;
    }
    [[nodiscard]] uint32_t root_index() const noexcept {
        return root_index_;
    }
    [[nodiscard]] const std::vector<BVHNode>& nodes() const noexcept {
        return nodes_;
    }

    [[nodiscard]] const AABB& root_bbox() const noexcept {
        static const AABB empty_bbox{};
        if (nodes_.empty() || root_index_ == 0xFFFFFFFF) {
            return empty_bbox;
        }
        return nodes_[root_index_].bbox;
    }

    void clear() noexcept {
        nodes_.clear();
        morton_keys_.clear();
        root_index_ = 0xFFFFFFFF;
        primitive_count_ = 0;
    }

private:
    alignas(64) std::vector<BVHNode> nodes_; // Flat array of 2N - 1 nodes
    std::vector<uint64_t> morton_keys_;      // 64-bit Morton codes
    uint32_t root_index_{0xFFFFFFFF};
    std::size_t primitive_count_{0};
};

} // namespace algoat::geometry

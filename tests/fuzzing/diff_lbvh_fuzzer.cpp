/**
 * @file diff_lbvh_fuzzer.cpp
 * @brief Differential Fuzzer for LBVH comparing against linear brute force.
 */

#include "algoat/geometry/lbvh.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace algoat::geometry;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    constexpr std::size_t box_bytes = 6 * sizeof(float); // min_x, min_y, min_z, max_x, max_y, max_z
    constexpr std::size_t header_bytes =
        6 * sizeof(float) + box_bytes; // ray (ox,oy,oz,dx,dy,dz) + query box
    if (size < header_bytes + box_bytes) {
        return 0;
    }

    float ray_params[6];
    std::memcpy(ray_params, data, 6 * sizeof(float));
    for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(ray_params[i])) {
            return 0;
        }
    }
    if (ray_params[3] == 0.0f && ray_params[4] == 0.0f && ray_params[5] == 0.0f) {
        ray_params[3] = 1.0f;
    }
    Ray ray{ray_params[0], ray_params[1], ray_params[2],
            ray_params[3], ray_params[4], ray_params[5]};

    float qbox_params[6];
    std::memcpy(qbox_params, data + 6 * sizeof(float), 6 * sizeof(float));
    for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(qbox_params[i])) {
            return 0;
        }
    }
    AABB query_window{
        std::min(qbox_params[0], qbox_params[3]), std::min(qbox_params[1], qbox_params[4]),
        std::min(qbox_params[2], qbox_params[5]), 0.0f,
        std::max(qbox_params[0], qbox_params[3]), std::max(qbox_params[1], qbox_params[4]),
        std::max(qbox_params[2], qbox_params[5]), 0.0f};

    const std::size_t remaining = size - header_bytes;
    std::size_t num_boxes = remaining / box_bytes;
    if (num_boxes > 150) {
        num_boxes = 150; // Bound size for fast fuzz iterations
    }

    std::vector<AABB> boxes(num_boxes);
    for (std::size_t i = 0; i < num_boxes; ++i) {
        float b[6];
        std::memcpy(b, data + header_bytes + (i * box_bytes), 6 * sizeof(float));
        for (int k = 0; k < 6; ++k) {
            if (!std::isfinite(b[k])) {
                return 0;
            }
        }
        boxes[i] = AABB{std::min(b[0], b[3]),         std::min(b[1], b[4]),
                        std::min(b[2], b[5]),         0.0f,
                        std::max(b[0], b[3]) + 0.01f, std::max(b[1], b[4]) + 0.01f,
                        std::max(b[2], b[5]) + 0.01f, 0.0f};
    }

    LBVH<AABB> bvh;
    bvh.build(boxes);

    if (bvh.leaf_count() != num_boxes) {
        __builtin_trap();
    }
    if (num_boxes > 0 && bvh.node_count() != (2 * num_boxes - 1)) {
        __builtin_trap();
    }

    // 1. Differential test query_box vs brute force
    auto actual_box = bvh.query_box(query_window);
    std::sort(actual_box.begin(), actual_box.end());

    std::vector<uint32_t> expected_box;
    for (uint32_t i = 0; i < num_boxes; ++i) {
        if (boxes[i].intersects(query_window)) {
            expected_box.push_back(i);
        }
    }
    std::sort(expected_box.begin(), expected_box.end());

    if (actual_box != expected_box) {
        __builtin_trap();
    }

    // 2. Differential test query_ray vs brute force
    std::optional<uint32_t> bf_hit = std::nullopt;
    float bf_min_t = std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < num_boxes; ++i) {
        float t = 0.0f;
        if (intersect_ray_aabb(ray, boxes[i], t)) {
            if (t < bf_min_t && t >= ray.t_min) {
                bf_min_t = t;
                bf_hit = i;
            }
        }
    }

    auto lbvh_hit =
        bvh.query_ray(ray, [&](const Ray& qray, uint32_t prim_id) -> std::optional<float> {
            float t = 0.0f;
            if (intersect_ray_aabb(qray, boxes[prim_id], t)) {
                return t;
            }
            return std::nullopt;
        });

    if (bf_hit.has_value()) {
        if (!lbvh_hit.has_value()) {
            __builtin_trap();
        }
        float lbvh_t = 0.0f;
        intersect_ray_aabb(ray, boxes[*lbvh_hit], lbvh_t);
        if (std::abs(lbvh_t - bf_min_t) > 1e-4f) {
            __builtin_trap();
        }
    } else {
        if (lbvh_hit.has_value()) {
            __builtin_trap();
        }
    }

    return 0;
}

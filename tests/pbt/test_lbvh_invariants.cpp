/**
 * @file test_lbvh_invariants.cpp
 * @brief RapidCheck Property-Based Tests for LBVH invariants.
 */

#include "algoat/geometry/lbvh.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::geometry;

static AABB make_box_from_raw(const std::vector<int>& raw) {
    float x1 = static_cast<float>(raw[0]) / 10.0f;
    float y1 = static_cast<float>(raw[1]) / 10.0f;
    float z1 = static_cast<float>(raw[2]) / 10.0f;
    float x2 = static_cast<float>(raw[3]) / 10.0f;
    float y2 = static_cast<float>(raw[4]) / 10.0f;
    float z2 = static_cast<float>(raw[5]) / 10.0f;

    return AABB{std::min(x1, x2),        std::min(y1, y2),        std::min(z1, z2),        0.0f,
                std::max(x1, x2) + 0.1f, std::max(y1, y2) + 0.1f, std::max(z1, z2) + 0.1f, 0.0f};
}

TEST(LBVHPBT, NodeCountAndBoundingInvariants) {
    rc::check("LBVH node count is exactly 2N - 1 and root encloses all primitives",
              [](const std::vector<std::vector<int>>& raw_boxes) {
                  RC_PRE(!raw_boxes.empty());

                  std::vector<AABB> boxes;
                  boxes.reserve(raw_boxes.size());
                  AABB scene_box;

                  for (const auto& raw : raw_boxes) {
                      if (raw.size() >= 6) {
                          AABB box = make_box_from_raw(raw);
                          boxes.push_back(box);
                          scene_box.expand(box);
                      }
                  }
                  RC_PRE(!boxes.empty());

                  LBVH<AABB> bvh;
                  bvh.build(boxes);

                  const std::size_t N = boxes.size();
                  RC_ASSERT(bvh.leaf_count() == N);
                  RC_ASSERT(bvh.node_count() == (2 * N - 1));

                  // Root AABB must match scene bounding box
                  RC_ASSERT(std::abs(bvh.root_bbox().min_x - scene_box.min_x) < 1e-4f);
                  RC_ASSERT(std::abs(bvh.root_bbox().min_y - scene_box.min_y) < 1e-4f);
                  RC_ASSERT(std::abs(bvh.root_bbox().min_z - scene_box.min_z) < 1e-4f);
                  RC_ASSERT(std::abs(bvh.root_bbox().max_x - scene_box.max_x) < 1e-4f);
                  RC_ASSERT(std::abs(bvh.root_bbox().max_y - scene_box.max_y) < 1e-4f);
                  RC_ASSERT(std::abs(bvh.root_bbox().max_z - scene_box.max_z) < 1e-4f);
              });
}

TEST(LBVHPBT, QueryBoxMatchesBruteForce) {
    rc::check("LBVH query_box returns exact set of overlapping primitives",
              [](const std::vector<std::vector<int>>& raw_boxes, const std::vector<int>& raw_q) {
                  RC_PRE(!raw_boxes.empty());
                  RC_PRE(raw_q.size() >= 6);

                  std::vector<AABB> boxes;
                  boxes.reserve(raw_boxes.size());
                  for (const auto& raw : raw_boxes) {
                      if (raw.size() >= 6) {
                          boxes.push_back(make_box_from_raw(raw));
                      }
                  }
                  RC_PRE(!boxes.empty());

                  float qx1 = static_cast<float>(raw_q[0]) / 10.0f;
                  float qy1 = static_cast<float>(raw_q[1]) / 10.0f;
                  float qz1 = static_cast<float>(raw_q[2]) / 10.0f;
                  float qx2 = static_cast<float>(raw_q[3]) / 10.0f;
                  float qy2 = static_cast<float>(raw_q[4]) / 10.0f;
                  float qz2 = static_cast<float>(raw_q[5]) / 10.0f;

                  AABB query_window{
                      std::min(qx1, qx2), std::min(qy1, qy2), std::min(qz1, qz2), 0.0f,
                      std::max(qx1, qx2), std::max(qy1, qy2), std::max(qz1, qz2), 0.0f};

                  LBVH<AABB> bvh;
                  bvh.build(boxes);

                  auto actual = bvh.query_box(query_window);
                  std::sort(actual.begin(), actual.end());

                  std::vector<uint32_t> expected;
                  for (uint32_t i = 0; i < boxes.size(); ++i) {
                      if (boxes[i].intersects(query_window)) {
                          expected.push_back(i);
                      }
                  }
                  std::sort(expected.begin(), expected.end());

                  RC_ASSERT(actual == expected);
              });
}

TEST(LBVHPBT, RayQueryMatchesBruteForce) {
    rc::check("LBVH query_ray finds the nearest primitive matching brute force",
              [](const std::vector<std::vector<int>>& raw_boxes, const std::vector<int>& raw_ray) {
                  RC_PRE(!raw_boxes.empty());
                  RC_PRE(raw_ray.size() >= 6);

                  std::vector<AABB> boxes;
                  boxes.reserve(raw_boxes.size());
                  for (const auto& raw : raw_boxes) {
                      if (raw.size() >= 6) {
                          boxes.push_back(make_box_from_raw(raw));
                      }
                  }
                  RC_PRE(!boxes.empty());

                  float ox = static_cast<float>(raw_ray[0]) / 10.0f;
                  float oy = static_cast<float>(raw_ray[1]) / 10.0f;
                  float oz = static_cast<float>(raw_ray[2]) / 10.0f;
                  float dx = static_cast<float>(raw_ray[3]) / 10.0f;
                  float dy = static_cast<float>(raw_ray[4]) / 10.0f;
                  float dz = static_cast<float>(raw_ray[5]) / 10.0f;
                  if (dx == 0.0f && dy == 0.0f && dz == 0.0f)
                      dx = 1.0f;

                  Ray ray{ox, oy, oz, dx, dy, dz};

                  LBVH<AABB> bvh;
                  bvh.build(boxes);

                  // Brute force
                  std::optional<uint32_t> bf_hit = std::nullopt;
                  float bf_min_t = std::numeric_limits<float>::infinity();
                  for (uint32_t i = 0; i < boxes.size(); ++i) {
                      float t = 0.0f;
                      if (intersect_ray_aabb(ray, boxes[i], t)) {
                          if (t < bf_min_t && t >= ray.t_min) {
                              bf_min_t = t;
                              bf_hit = i;
                          }
                      }
                  }

                  auto lbvh_hit = bvh.query_ray(
                      ray, [&](const Ray& qray, uint32_t prim_id) -> std::optional<float> {
                          float t = 0.0f;
                          if (intersect_ray_aabb(qray, boxes[prim_id], t)) {
                              return t;
                          }
                          return std::nullopt;
                      });

                  if (bf_hit.has_value()) {
                      RC_ASSERT(lbvh_hit.has_value());
                      float lbvh_t = 0.0f;
                      intersect_ray_aabb(ray, boxes[*lbvh_hit], lbvh_t);
                      RC_ASSERT(std::abs(lbvh_t - bf_min_t) < 1e-4f);
                  } else {
                      RC_ASSERT(!lbvh_hit.has_value());
                  }
              });
}

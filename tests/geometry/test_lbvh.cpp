/**
 * @file test_lbvh.cpp
 * @brief Unit tests for Linear Bounding Volume Hierarchy (LBVH).
 */

#include "algoat/geometry/lbvh.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

namespace algoat::geometry::testing {

TEST(LBVHTest, AABBInitializationAndExpansion) {
    AABB box;
    EXPECT_TRUE(std::isinf(box.min_x));
    EXPECT_TRUE(std::isinf(box.min_y));
    EXPECT_TRUE(std::isinf(box.min_z));
    EXPECT_LT(box.max_x, box.min_x);

    AABB point_box{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f};
    box.expand(point_box);

    EXPECT_FLOAT_EQ(box.min_x, 0.0f);
    EXPECT_FLOAT_EQ(box.min_y, 0.0f);
    EXPECT_FLOAT_EQ(box.min_z, 0.0f);
    EXPECT_FLOAT_EQ(box.max_x, 1.0f);
    EXPECT_FLOAT_EQ(box.max_y, 2.0f);
    EXPECT_FLOAT_EQ(box.max_z, 3.0f);

    auto center = box.center();
    EXPECT_FLOAT_EQ(center[0], 0.5f);
    EXPECT_FLOAT_EQ(center[1], 1.0f);
    EXPECT_FLOAT_EQ(center[2], 1.5f);
}

TEST(LBVHTest, AABBIntersectsPredicate) {
    AABB b1{0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f, 0.0f};
    AABB b2{1.0f, 1.0f, 1.0f, 0.0f, 3.0f, 3.0f, 3.0f, 0.0f};
    AABB b3{3.0f, 3.0f, 3.0f, 0.0f, 5.0f, 5.0f, 5.0f, 0.0f};

    EXPECT_TRUE(b1.intersects(b2));
    EXPECT_TRUE(b2.intersects(b1));
    EXPECT_FALSE(b1.intersects(b3));
    EXPECT_FALSE(b3.intersects(b1));
}

TEST(LBVHTest, RayConstructionAndInverseDirection) {
    Ray r1{1.0f, 2.0f, 3.0f, 2.0f, -4.0f, 0.0f};
    EXPECT_FLOAT_EQ(r1.origin_x, 1.0f);
    EXPECT_FLOAT_EQ(r1.origin_y, 2.0f);
    EXPECT_FLOAT_EQ(r1.origin_z, 3.0f);
    EXPECT_FLOAT_EQ(r1.dir_x, 2.0f);
    EXPECT_FLOAT_EQ(r1.dir_y, -4.0f);
    EXPECT_FLOAT_EQ(r1.dir_z, 0.0f);

    EXPECT_FLOAT_EQ(r1.inv_dir_x, 0.5f);
    EXPECT_FLOAT_EQ(r1.inv_dir_y, -0.25f);
    EXPECT_TRUE(std::isinf(r1.inv_dir_z));
    EXPECT_FLOAT_EQ(r1.t_min, 0.0f);
    EXPECT_TRUE(std::isinf(r1.t_max));
}

TEST(LBVHTest, Morton3DEncodingAndEquivalence) {
    // Test known bit patterns
    EXPECT_EQ(morton3d_encode_fallback(0, 0, 0), 0ULL);
    EXPECT_EQ(morton3d_encode_fallback(1, 0, 0), 1ULL);
    EXPECT_EQ(morton3d_encode_fallback(0, 1, 0), 2ULL);
    EXPECT_EQ(morton3d_encode_fallback(0, 0, 1), 4ULL);
    EXPECT_EQ(morton3d_encode_fallback(1, 1, 1), 7ULL);

    // Test equivalence between BMI2 and fallback
    for (uint32_t i = 0; i < 1000; ++i) {
        uint32_t x = (i * 2654435761u) & 0x1FFFFFu;
        uint32_t y = (i * 2246822519u) & 0x1FFFFFu;
        uint32_t z = (i * 3266489917u) & 0x1FFFFFu;
        EXPECT_EQ(morton3d_encode(x, y, z), morton3d_encode_fallback(x, y, z));
    }
}

TEST(LBVHTest, RadixSort16BitMatchesStdSort) {
    constexpr std::size_t N = 50000;
    std::vector<MortonItem> items(N);
    for (std::size_t i = 0; i < N; ++i) {
        uint64_t code = ((static_cast<uint64_t>(i * 6364136223846793005ULL) >> 1)); // 63 bits
        items[i] = MortonItem{code, static_cast<uint32_t>(i), 0};
    }

    std::vector<MortonItem> expected = items;
    std::stable_sort(expected.begin(), expected.end(),
                     [](const auto& a, const auto& b) { return a.code < b.code; });

    radix_sort_morton(items);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_EQ(items[i].code, expected[i].code);
        EXPECT_EQ(items[i].index, expected[i].index);
    }
}

TEST(LBVHTest, TreeTopologyAndBoundingBoxHierarchy) {
    std::vector<AABB> boxes;
    // Create 100 boxes in 3D grid
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 4; ++y) {
            for (int z = 0; z < 5; ++z) {
                float fx = static_cast<float>(x) * 2.0f;
                float fy = static_cast<float>(y) * 2.0f;
                float fz = static_cast<float>(z) * 2.0f;
                boxes.push_back(AABB{fx, fy, fz, 0.0f, fx + 1.0f, fy + 1.0f, fz + 1.0f, 0.0f});
            }
        }
    }
    const std::size_t N = boxes.size();
    EXPECT_EQ(N, 100);

    LBVH<AABB> bvh;
    bvh.build(boxes);

    EXPECT_EQ(bvh.leaf_count(), N);
    EXPECT_EQ(bvh.node_count(), 2 * N - 1);

    // Verify root bounding box covers all boxes
    AABB scene_box;
    for (const auto& b : boxes) {
        scene_box.expand(b);
    }
    EXPECT_FLOAT_EQ(bvh.root_bbox().min_x, scene_box.min_x);
    EXPECT_FLOAT_EQ(bvh.root_bbox().min_y, scene_box.min_y);
    EXPECT_FLOAT_EQ(bvh.root_bbox().min_z, scene_box.min_z);
    EXPECT_FLOAT_EQ(bvh.root_bbox().max_x, scene_box.max_x);
    EXPECT_FLOAT_EQ(bvh.root_bbox().max_y, scene_box.max_y);
    EXPECT_FLOAT_EQ(bvh.root_bbox().max_z, scene_box.max_z);
}

TEST(LBVHTest, QueryBoxFindsExactOverlaps) {
    std::vector<AABB> boxes;
    for (int i = 0; i < 50; ++i) {
        float f = static_cast<float>(i) * 2.0f;
        boxes.push_back(AABB{f, 0.0f, 0.0f, 0.0f, f + 1.0f, 1.0f, 1.0f, 0.0f});
    }

    LBVH<AABB> bvh;
    bvh.build(boxes);

    // Query a box spanning [3.5, 7.5] along X: strictly overlaps boxes 2 (at 4..5) and 3 (at 6..7)
    AABB query_window{3.5f, -0.5f, -0.5f, 0.0f, 7.5f, 1.5f, 1.5f, 0.0f};
    auto hits = bvh.query_box(query_window);

    std::sort(hits.begin(), hits.end());
    std::vector<uint32_t> expected = {2, 3};
    EXPECT_EQ(hits, expected);
}

TEST(LBVHTest, QueryRayFindsNearestPrimitive) {
    // 10 boxes along the X axis
    std::vector<AABB> boxes;
    for (int i = 0; i < 10; ++i) {
        float fx = static_cast<float>(i) * 5.0f; // 0, 5, 10, ..., 45
        boxes.push_back(AABB{fx, -1.0f, -1.0f, 0.0f, fx + 2.0f, 1.0f, 1.0f, 0.0f});
    }

    LBVH<AABB> bvh;
    bvh.build(boxes);

    // Ray shooting along +X from -10.0: should hit box 0 first
    Ray r1{-10.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    auto hit0 = bvh.query_ray(r1, [&](const Ray& ray, uint32_t prim_id) -> std::optional<float> {
        float t_hit = 0.0f;
        if (intersect_ray_aabb(ray, boxes[prim_id], t_hit)) {
            return t_hit;
        }
        return std::nullopt;
    });
    ASSERT_TRUE(hit0.has_value());
    EXPECT_EQ(*hit0, 0u);

    // Ray shooting along -X from 50.0: should hit box 9 first (at 45..47)
    Ray r2{50.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f};
    auto hit9 = bvh.query_ray(r2, [&](const Ray& ray, uint32_t prim_id) -> std::optional<float> {
        float t_hit = 0.0f;
        if (intersect_ray_aabb(ray, boxes[prim_id], t_hit)) {
            return t_hit;
        }
        return std::nullopt;
    });
    ASSERT_TRUE(hit9.has_value());
    EXPECT_EQ(*hit9, 9u);

    // Ray shooting along Y from above: misses all boxes
    Ray r_miss{2.5f, 10.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    auto hit_none =
        bvh.query_ray(r_miss, [&](const Ray& ray, uint32_t prim_id) -> std::optional<float> {
            float t_hit = 0.0f;
            if (intersect_ray_aabb(ray, boxes[prim_id], t_hit)) {
                return t_hit;
            }
            return std::nullopt;
        });
    EXPECT_FALSE(hit_none.has_value());
}

TEST(LBVHTest, EmptyAndSinglePrimitive) {
    LBVH<AABB> bvh_empty;
    std::vector<AABB> empty;
    bvh_empty.build(empty);
    EXPECT_EQ(bvh_empty.node_count(), 0);
    EXPECT_EQ(bvh_empty.leaf_count(), 0);
    EXPECT_FALSE(
        bvh_empty.query_ray(Ray{0, 0, 0, 1, 0, 0}, [](const Ray&, uint32_t) { return true; })
            .has_value());
    EXPECT_TRUE(bvh_empty.query_box(AABB{0, 0, 0, 0, 1, 1, 1, 0}).empty());

    LBVH<AABB> bvh_single;
    std::vector<AABB> single = {AABB{1.0f, 1.0f, 1.0f, 0.0f, 2.0f, 2.0f, 2.0f, 0.0f}};
    bvh_single.build(single);
    EXPECT_EQ(bvh_single.node_count(), 1);
    EXPECT_EQ(bvh_single.leaf_count(), 1);

    Ray r{0.0f, 1.5f, 1.5f, 1.0f, 0.0f, 0.0f};
    auto hit =
        bvh_single.query_ray(r, [&](const Ray& ray, uint32_t prim_id) -> std::optional<float> {
            float t_hit = 0.0f;
            if (intersect_ray_aabb(ray, single[prim_id], t_hit))
                return t_hit;
            return std::nullopt;
        });
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(*hit, 0u);
}

TEST(LBVHTest, DegenerateIdenticalCentroidsAndCollinear) {
    // 20 primitives with the exact same coordinates
    std::vector<AABB> identical_boxes(20, AABB{5.0f, 5.0f, 5.0f, 0.0f, 6.0f, 6.0f, 6.0f, 0.0f});
    LBVH<AABB> bvh;
    bvh.build(identical_boxes);

    EXPECT_EQ(bvh.leaf_count(), 20);
    EXPECT_EQ(bvh.node_count(), 2 * 20 - 1);

    auto hits = bvh.query_box(AABB{4.0f, 4.0f, 4.0f, 0.0f, 7.0f, 7.0f, 7.0f, 0.0f});
    EXPECT_EQ(hits.size(), 20);

    // 50 collinear point boxes (extent = 0)
    std::vector<AABB> points;
    for (int i = 0; i < 50; ++i) {
        float z = static_cast<float>(i);
        points.push_back(AABB{0.0f, 0.0f, z, 0.0f, 0.0f, 0.0f, z, 0.0f});
    }
    LBVH<AABB> bvh_pts;
    bvh_pts.build(points);
    EXPECT_EQ(bvh_pts.leaf_count(), 50);
    EXPECT_EQ(bvh_pts.node_count(), 2 * 50 - 1);
}

TEST(LBVHTest, RayQueryMatchesBruteForceRandomGeometry) {
    // 500 random spheres/boxes
    std::vector<AABB> boxes;
    for (int i = 0; i < 500; ++i) {
        float x = static_cast<float>((i * 17) % 100);
        float y = static_cast<float>((i * 31) % 100);
        float z = static_cast<float>((i * 47) % 100);
        boxes.push_back(AABB{x, y, z, 0.0f, x + 2.0f, y + 2.0f, z + 2.0f, 0.0f});
    }

    LBVH<AABB> bvh;
    bvh.build(boxes);

    // Test 1000 randomized rays
    for (int r = 0; r < 1000; ++r) {
        float ox = static_cast<float>((r * 13) % 120) - 10.0f;
        float oy = static_cast<float>((r * 29) % 120) - 10.0f;
        float oz = static_cast<float>((r * 41) % 120) - 10.0f;

        float dx = static_cast<float>((r * 7) % 21) - 10.0f;
        float dy = static_cast<float>((r * 19) % 21) - 10.0f;
        float dz = static_cast<float>((r * 23) % 21) - 10.0f;
        if (dx == 0.0f && dy == 0.0f && dz == 0.0f)
            dx = 1.0f;

        Ray ray{ox, oy, oz, dx, dy, dz};

        // Exhaustive brute force
        std::optional<uint32_t> bf_best = std::nullopt;
        float bf_min_t = std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < boxes.size(); ++i) {
            float t_hit = 0.0f;
            if (intersect_ray_aabb(ray, boxes[i], t_hit)) {
                if (t_hit < bf_min_t && t_hit >= ray.t_min) {
                    bf_min_t = t_hit;
                    bf_best = i;
                }
            }
        }

        // LBVH query
        auto lbvh_hit =
            bvh.query_ray(ray, [&](const Ray& qray, uint32_t prim_id) -> std::optional<float> {
                float t_hit = 0.0f;
                if (intersect_ray_aabb(qray, boxes[prim_id], t_hit)) {
                    return t_hit;
                }
                return std::nullopt;
            });

        if (bf_best.has_value()) {
            ASSERT_TRUE(lbvh_hit.has_value());
            // Since multiple boxes might share the same min t_hit, verify t_hit matches
            float t_lbvh = 0.0f;
            intersect_ray_aabb(ray, boxes[*lbvh_hit], t_lbvh);
            EXPECT_NEAR(t_lbvh, bf_min_t, 1e-4f);
        } else {
            EXPECT_FALSE(lbvh_hit.has_value());
        }
    }
}

#if defined(__AVX2__)
TEST(LBVHTest, AVX2FourBoxSlabTest) {
    Ray ray{0.0f, 0.0f, -10.0f, 0.0f, 0.0f, 1.0f}; // Ray along +Z

    // 4 boxes: box 0, 1, 2 intersect; box 3 is completely off to the side
    AABB b0{-1.0f, -1.0f, 10.0f, 0.0f, 1.0f, 1.0f, 12.0f, 0.0f};
    AABB b1{-0.5f, -0.5f, 5.0f, 0.0f, 0.5f, 0.5f, 7.0f, 0.0f};
    AABB b2{-2.0f, -2.0f, 20.0f, 0.0f, 2.0f, 2.0f, 25.0f, 0.0f};
    AABB b3{50.0f, 50.0f, 5.0f, 0.0f, 60.0f, 60.0f, 10.0f, 0.0f};

    __m256 min_x = _mm256_set_ps(0, 0, 0, 0, b3.min_x, b2.min_x, b1.min_x, b0.min_x);
    __m256 min_y = _mm256_set_ps(0, 0, 0, 0, b3.min_y, b2.min_y, b1.min_y, b0.min_y);
    __m256 min_z = _mm256_set_ps(0, 0, 0, 0, b3.min_z, b2.min_z, b1.min_z, b0.min_z);
    __m256 max_x = _mm256_set_ps(0, 0, 0, 0, b3.max_x, b2.max_x, b1.max_x, b0.max_x);
    __m256 max_y = _mm256_set_ps(0, 0, 0, 0, b3.max_y, b2.max_y, b1.max_y, b0.max_y);
    __m256 max_z = _mm256_set_ps(0, 0, 0, 0, b3.max_z, b2.max_z, b1.max_z, b0.max_z);

    alignas(32) float t_nears[8];
    int hit_mask =
        detail::intersect_ray_4boxes_avx2(ray, min_x, min_y, min_z, max_x, max_y, max_z, t_nears);

    EXPECT_TRUE((hit_mask & 1) != 0);  // b0 hit
    EXPECT_TRUE((hit_mask & 2) != 0);  // b1 hit
    EXPECT_TRUE((hit_mask & 4) != 0);  // b2 hit
    EXPECT_FALSE((hit_mask & 8) != 0); // b3 missed

    EXPECT_FLOAT_EQ(t_nears[0], 20.0f); // from -10 to 10 is 20
    EXPECT_FLOAT_EQ(t_nears[1], 15.0f); // from -10 to 5 is 15
    EXPECT_FLOAT_EQ(t_nears[2], 30.0f); // from -10 to 20 is 30
}
#endif

TEST(LBVHTest, LargeScaleMeshRayIntersectionCorrectness) {
    constexpr std::size_t N = 250000;
    std::vector<AABB> mesh_triangles(N);

    // Procedurally generate a complex surface mesh (helicoid/torus)
    for (std::size_t i = 0; i < N; ++i) {
        float u = static_cast<float>(i % 500) * 0.1f;
        float v = static_cast<float>(i / 500) * 0.1f;
        float x = (20.0f + 5.0f * std::cos(v)) * std::cos(u);
        float y = (20.0f + 5.0f * std::cos(v)) * std::sin(u);
        float z = 5.0f * std::sin(v) + 0.05f * u;
        mesh_triangles[i] = AABB{x, y, z, 0.0f, x + 0.2f, y + 0.2f, z + 0.2f, 0.0f};
    }

    LBVH<AABB> bvh;
    bvh.build(mesh_triangles);
    EXPECT_EQ(bvh.leaf_count(), N);
    EXPECT_EQ(bvh.node_count(), 2 * N - 1);

    // Test randomized rays across the complex mesh verifying against brute force on sample rays
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> pos_dist(-40.0f, 40.0f);
    std::uniform_real_distribution<float> dir_dist(-1.0f, 1.0f);

    for (int r = 0; r < 500; ++r) {
        Ray ray{pos_dist(rng), pos_dist(rng), pos_dist(rng),
                dir_dist(rng), dir_dist(rng), dir_dist(rng)};

        std::optional<uint32_t> bf_best = std::nullopt;
        float bf_min_t = std::numeric_limits<float>::infinity();

        for (uint32_t i = 0; i < N; ++i) {
            float t = 0.0f;
            if (intersect_ray_aabb(ray, mesh_triangles[i], t)) {
                if (t < bf_min_t && t >= ray.t_min) {
                    bf_min_t = t;
                    bf_best = i;
                }
            }
        }

        auto lbvh_hit =
            bvh.query_ray(ray, [&](const Ray& qray, uint32_t prim_id) -> std::optional<float> {
                float t = 0.0f;
                if (intersect_ray_aabb(qray, mesh_triangles[prim_id], t)) {
                    return t;
                }
                return std::nullopt;
            });

        if (bf_best.has_value()) {
            ASSERT_TRUE(lbvh_hit.has_value());
            float lbvh_t = 0.0f;
            intersect_ray_aabb(ray, mesh_triangles[*lbvh_hit], lbvh_t);
            EXPECT_NEAR(lbvh_t, bf_min_t, 1e-4f);
        } else {
            EXPECT_FALSE(lbvh_hit.has_value());
        }
    }
}

} // namespace algoat::geometry::testing

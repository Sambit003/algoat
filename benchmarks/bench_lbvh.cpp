/**
 * @file bench_lbvh.cpp
 * @brief Google Benchmark suite for Linear Bounding Volume Hierarchy (LBVH).
 */

#include "algoat/geometry/lbvh.hpp"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using namespace algoat::geometry;

static std::vector<AABB> generate_random_boxes(std::size_t count, float min_val = -100.0f,
                                               float max_val = 100.0f, float max_size = 2.0f) {
    std::vector<AABB> boxes(count);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> pos_dist(min_val, max_val);
    std::uniform_real_distribution<float> size_dist(0.1f, max_size);

    for (std::size_t i = 0; i < count; ++i) {
        float x = pos_dist(rng);
        float y = pos_dist(rng);
        float z = pos_dist(rng);
        float sx = size_dist(rng);
        float sy = size_dist(rng);
        float sz = size_dist(rng);
        boxes[i] = AABB{x, y, z, 0.0f, x + sx, y + sy, z + sz, 0.0f};
    }
    return boxes;
}

static std::vector<Ray> generate_random_rays(std::size_t count, float min_val = -120.0f,
                                             float max_val = 120.0f) {
    std::vector<Ray> rays(count);
    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<float> pos_dist(min_val, max_val);
    std::uniform_real_distribution<float> dir_dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < count; ++i) {
        float ox = pos_dist(rng);
        float oy = pos_dist(rng);
        float oz = pos_dist(rng);
        float dx = dir_dist(rng);
        float dy = dir_dist(rng);
        float dz = dir_dist(rng);
        if (dx == 0.0f && dy == 0.0f && dz == 0.0f) {
            dx = 1.0f;
        }
        rays[i] = Ray{ox, oy, oz, dx, dy, dz};
    }
    return rays;
}

static void BM_Algoat_LBVH_Build(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto boxes = generate_random_boxes(n);

    for (auto _ : state) {
        LBVH<AABB> bvh;
        bvh.build(boxes);
        benchmark::DoNotOptimize(bvh.node_count());
    }
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Algoat_LBVH_Build)->RangeMultiplier(10)->Range(10000, 1000000)->Complexity();

static void BM_BruteForce_RayQuery(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto boxes = generate_random_boxes(n);
    const auto rays = generate_random_rays(100);

    std::size_t r_idx = 0;
    for (auto _ : state) {
        const auto& ray = rays[r_idx];
        std::optional<uint32_t> best_hit = std::nullopt;
        float min_t = std::numeric_limits<float>::infinity();

        for (uint32_t i = 0; i < n; ++i) {
            float t = 0.0f;
            if (intersect_ray_aabb(ray, boxes[i], t)) {
                if (t < min_t && t >= ray.t_min) {
                    min_t = t;
                    best_hit = i;
                }
            }
        }
        benchmark::DoNotOptimize(best_hit);
        r_idx = (r_idx + 1) % rays.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_BruteForce_RayQuery)->RangeMultiplier(10)->Range(1000, 50000)->Complexity();

static void BM_Algoat_LBVH_RayQuery(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto boxes = generate_random_boxes(n);
    LBVH<AABB> bvh;
    bvh.build(boxes);
    const auto rays = generate_random_rays(10000);

    std::size_t r_idx = 0;
    for (auto _ : state) {
        const auto& ray = rays[r_idx];
        auto hit =
            bvh.query_ray(ray, [&](const Ray& qray, uint32_t prim_id) -> std::optional<float> {
                float t = 0.0f;
                if (intersect_ray_aabb(qray, boxes[prim_id], t)) {
                    return t;
                }
                return std::nullopt;
            });
        benchmark::DoNotOptimize(hit);
        r_idx = (r_idx + 1) % rays.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Algoat_LBVH_RayQuery)->RangeMultiplier(10)->Range(10000, 1000000)->Complexity();

static void BM_Algoat_LBVH_BoxQuery(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto boxes = generate_random_boxes(n);
    LBVH<AABB> bvh;
    bvh.build(boxes);

    AABB query_window{-10.0f, -10.0f, -10.0f, 0.0f, 10.0f, 10.0f, 10.0f, 0.0f};

    for (auto _ : state) {
        auto hits = bvh.query_box(query_window);
        benchmark::DoNotOptimize(hits.data());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Algoat_LBVH_BoxQuery)->RangeMultiplier(10)->Range(10000, 1000000)->Complexity();

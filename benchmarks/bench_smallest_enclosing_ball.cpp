/**
 * @file bench_smallest_enclosing_ball.cpp
 * @brief Google Benchmark suite for Welzl's Smallest Enclosing Sphere (2D and 3D).
 */

#include "algoat/geometry/smallest_enclosing_ball.hpp"

#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

using namespace algoat::geometry;

namespace {

std::vector<std::array<double, 3>> generate_uniform_3d_points(std::size_t n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<std::array<double, 3>> pts;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts.push_back({dist(rng), dist(rng), dist(rng)});
    }
    return pts;
}

std::vector<std::array<double, 3>> generate_sphere_shell_3d_points(std::size_t n,
                                                                   uint64_t seed = 99) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::uniform_real_distribution<double> rad_jit(0.95, 1.0);
    std::vector<std::array<double, 3>> pts;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = norm(rng);
        const double y = norm(rng);
        const double z = norm(rng);
        const double inv = rad_jit(rng) / std::sqrt(x * x + y * y + z * z + 1e-18);
        pts.push_back({x * inv, y * inv, z * inv});
    }
    return pts;
}

std::vector<std::array<double, 2>> generate_uniform_2d_points(std::size_t n, uint64_t seed = 77) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<std::array<double, 2>> pts;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts.push_back({dist(rng), dist(rng)});
    }
    return pts;
}

void BM_SmallestEnclosingBall_3D_1M(benchmark::State& state) {
    constexpr std::size_t kN = 1000000;
    const auto pts = generate_uniform_3d_points(kN);
    double r = 0.0;
    for (auto _ : state) {
        auto res = smallest_enclosing_ball(std::span<const std::array<double, 3>>{pts});
        r = res.radius;
        benchmark::DoNotOptimize(res);
    }
    state.counters["N"] = static_cast<double>(kN);
    state.counters["radius"] = r;
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));
}

void BM_SmallestEnclosingBall_3D_BoundarySphere_1M(benchmark::State& state) {
    constexpr std::size_t kN = 1000000;
    const auto pts = generate_sphere_shell_3d_points(kN);
    double r = 0.0;
    for (auto _ : state) {
        auto res = smallest_enclosing_ball(std::span<const std::array<double, 3>>{pts});
        r = res.radius;
        benchmark::DoNotOptimize(res);
    }
    state.counters["N"] = static_cast<double>(kN);
    state.counters["radius"] = r;
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));
}

void BM_SmallestEnclosingCircle_2D_1M(benchmark::State& state) {
    constexpr std::size_t kN = 1000000;
    const auto pts = generate_uniform_2d_points(kN);
    double r = 0.0;
    for (auto _ : state) {
        auto res = smallest_enclosing_circle(std::span<const std::array<double, 2>>{pts});
        r = res.radius;
        benchmark::DoNotOptimize(res);
    }
    state.counters["N"] = static_cast<double>(kN);
    state.counters["radius"] = r;
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));
}

void BM_SmallestEnclosingBall_3D_Scaling(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto pts = generate_uniform_3d_points(n);
    for (auto _ : state) {
        auto res = smallest_enclosing_ball(std::span<const std::array<double, 3>>{pts});
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}

} // namespace

BENCHMARK(BM_SmallestEnclosingBall_3D_1M)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SmallestEnclosingBall_3D_BoundarySphere_1M)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SmallestEnclosingCircle_2D_1M)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SmallestEnclosingBall_3D_Scaling)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMillisecond);

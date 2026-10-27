/**
 * @file bench_delaunay.cpp
 * @brief Google Benchmark suite for 2D Delaunay Triangulation and Voronoi Dual.
 */

#include "algoat/geometry/delaunay.hpp"

#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace algoat::geometry;

static std::vector<Point2D> generate_random_points(std::size_t n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<Point2D> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = {dist(rng), dist(rng)};
    }
    return pts;
}

static void BM_Delaunay_Build(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto points = generate_random_points(n);

    for (auto _ : state) {
        DelaunayTriangulation2D dt;
        dt.build(points);
        benchmark::DoNotOptimize(dt.triangle_count());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_Delaunay_Build)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(500000)
    ->Unit(benchmark::kMillisecond);

static void BM_Delaunay_LocatePoint(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto points = generate_random_points(n);

    DelaunayTriangulation2D dt;
    dt.build(points);

    const auto queries = generate_random_points(1000, 1337);
    std::size_t q_idx = 0;

    for (auto _ : state) {
        const auto& q = queries[q_idx % queries.size()];
        auto loc = dt.locate_point(q);
        benchmark::DoNotOptimize(loc);
        ++q_idx;
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_Delaunay_LocatePoint)->Arg(10000)->Arg(100000)->Unit(benchmark::kNanosecond);

static void BM_Delaunay_NearestSite(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto points = generate_random_points(n);

    DelaunayTriangulation2D dt;
    dt.build(points);

    const auto queries = generate_random_points(1000, 9999);
    std::size_t q_idx = 0;

    for (auto _ : state) {
        const auto& q = queries[q_idx % queries.size()];
        uint32_t site = dt.nearest_site(q);
        benchmark::DoNotOptimize(site);
        ++q_idx;
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_Delaunay_NearestSite)->Arg(10000)->Arg(100000)->Unit(benchmark::kNanosecond);

static void BM_Voronoi_Dual(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto points = generate_random_points(n);

    DelaunayTriangulation2D dt;
    dt.build(points);

    for (auto _ : state) {
        auto cells = dt.compute_voronoi_cells();
        benchmark::DoNotOptimize(cells.size());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_Voronoi_Dual)->Arg(1000)->Arg(10000)->Unit(benchmark::kMillisecond);

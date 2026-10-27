/**
 * @file bench_polygon.cpp
 * @brief Google Benchmark suite for SIMD-Vectorized Point-in-Polygon (PIP) classification.
 */

#include "algoat/geometry/polygon.hpp"
#include "algoat/geometry/predicates.hpp"

#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <vector>

using namespace algoat::geometry;

static std::vector<Point2D> generate_star_polygon(std::size_t n, double radius = 500.0) {
    std::vector<Point2D> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        double theta = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n);
        // Perturb radius slightly to create non-convex concave features
        double r = radius * (1.0 + 0.25 * std::sin(7.0 * theta) + 0.1 * std::cos(13.0 * theta));
        pts[i] = {r * std::cos(theta), r * std::sin(theta)};
    }
    return pts;
}

static std::vector<Point2D> generate_query_points(std::size_t m, double bound = 700.0,
                                                  uint64_t seed = 12345) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-bound, bound);
    std::vector<Point2D> queries(m);
    for (std::size_t i = 0; i < m; ++i) {
        queries[i] = {dist(rng), dist(rng)};
    }
    return queries;
}

static void BM_Polygon_Scalar_Winding(benchmark::State& state) {
    const auto n_edges = static_cast<std::size_t>(state.range(0));
    const auto poly_pts = generate_star_polygon(n_edges);
    const Polygon poly(poly_pts);

    constexpr std::size_t kQueryCount = 10000;
    const auto queries = generate_query_points(kQueryCount);

    for (auto _ : state) {
        std::size_t inside_count = 0;
        for (const auto& q : queries) {
            auto res = poly.contains_scalar(q, InclusionRule::NonZeroWinding);
            inside_count += (res == PolygonInclusion::Inside);
        }
        benchmark::DoNotOptimize(inside_count);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * kQueryCount);
    state.counters["edges_per_sec"] =
        benchmark::Counter(static_cast<double>(state.iterations() * kQueryCount * n_edges),
                           benchmark::Counter::kIsRate);
}

static void BM_Polygon_Scalar_EvenOdd(benchmark::State& state) {
    const auto n_edges = static_cast<std::size_t>(state.range(0));
    const auto poly_pts = generate_star_polygon(n_edges);
    const Polygon poly(poly_pts);

    constexpr std::size_t kQueryCount = 10000;
    const auto queries = generate_query_points(kQueryCount);

    for (auto _ : state) {
        std::size_t inside_count = 0;
        for (const auto& q : queries) {
            auto res = poly.contains_scalar(q, InclusionRule::EvenOdd);
            inside_count += (res == PolygonInclusion::Inside);
        }
        benchmark::DoNotOptimize(inside_count);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * kQueryCount);
    state.counters["edges_per_sec"] =
        benchmark::Counter(static_cast<double>(state.iterations() * kQueryCount * n_edges),
                           benchmark::Counter::kIsRate);
}

static void BM_Polygon_SIMD_Batch_EvenOdd(benchmark::State& state) {
    const auto n_edges = static_cast<std::size_t>(state.range(0));
    const auto poly_pts = generate_star_polygon(n_edges);
    const Polygon poly(poly_pts);

    constexpr std::size_t kQueryCount = 10000;
    const auto queries = generate_query_points(kQueryCount);
    std::vector<PolygonInclusion> results(kQueryCount);

    for (auto _ : state) {
        poly.batch_contains(queries, results, InclusionRule::EvenOdd);
        benchmark::DoNotOptimize(results.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * kQueryCount);
    state.counters["edges_per_sec"] =
        benchmark::Counter(static_cast<double>(state.iterations() * kQueryCount * n_edges),
                           benchmark::Counter::kIsRate);
}

static void BM_Polygon_SIMD_Batch_Winding(benchmark::State& state) {
    const auto n_edges = static_cast<std::size_t>(state.range(0));
    const auto poly_pts = generate_star_polygon(n_edges);
    const Polygon poly(poly_pts);

    constexpr std::size_t kQueryCount = 10000;
    const auto queries = generate_query_points(kQueryCount);
    std::vector<PolygonInclusion> results(kQueryCount);

    for (auto _ : state) {
        poly.batch_contains(queries, results, InclusionRule::NonZeroWinding);
        benchmark::DoNotOptimize(results.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * kQueryCount);
    state.counters["edges_per_sec"] =
        benchmark::Counter(static_cast<double>(state.iterations() * kQueryCount * n_edges),
                           benchmark::Counter::kIsRate);
}

// GIS/Geospatial streaming benchmark with spatial rejection
static void BM_Polygon_SIMD_Batch_Streaming_GIS(benchmark::State& state) {
    const auto n_edges = static_cast<std::size_t>(state.range(0));
    const auto poly_pts = generate_star_polygon(n_edges);
    const Polygon poly(poly_pts);

    constexpr std::size_t kQueryCount = 100000;
    // Map tile / spatial domain 14x larger than polygon extent (typical GIS viewport/geofence
    // query)
    const auto queries = generate_query_points(kQueryCount, 7000.0);
    std::vector<PolygonInclusion> results(kQueryCount);

    for (auto _ : state) {
        poly.batch_contains(queries, results, InclusionRule::NonZeroWinding);
        benchmark::DoNotOptimize(results.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * kQueryCount);
}

BENCHMARK(BM_Polygon_Scalar_Winding)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Polygon_Scalar_EvenOdd)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Polygon_SIMD_Batch_EvenOdd)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Polygon_SIMD_Batch_Winding)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Polygon_SIMD_Batch_Streaming_GIS)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

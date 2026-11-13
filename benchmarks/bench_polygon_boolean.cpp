/**
 * @file bench_polygon_boolean.cpp
 * @brief Google Benchmark suite for Martinez-Rueda-Feito Polygon Boolean Operations.
 */

#include "algoat/geometry/polygon_boolean.hpp"
#include "algoat/geometry/predicates.hpp"

#include <benchmark/benchmark.h>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using namespace algoat::geometry;

namespace {

BooleanPolygon<Point2D> make_jittered_star(std::size_t n, double cx, double cy, double r_base,
                                           uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> jitter(0.7, 1.3);
    BooleanPolygon<Point2D> p;
    p.outer_ring.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double ang =
            (2.0 * std::numbers::pi * static_cast<double>(i)) / static_cast<double>(n);
        const double r = r_base * jitter(rng);
        p.outer_ring.push_back({cx + r * std::cos(ang), cy + r * std::sin(ang)});
    }
    return p;
}

BooleanPolygon<Point2D> make_horizontal_comb(std::size_t teeth) {
    BooleanPolygon<Point2D> p;
    const double h = static_cast<double>(teeth) * 2.0 + 1.0;
    p.outer_ring.push_back({-2.0, 0.0});
    p.outer_ring.push_back({-2.0, h});
    p.outer_ring.push_back({0.0, h});
    for (std::size_t i = teeth; i-- > 0;) {
        const double y_top = static_cast<double>(i) * 2.0 + 1.5;
        const double y_bot = static_cast<double>(i) * 2.0 + 0.5;
        p.outer_ring.push_back({10.0, y_top});
        p.outer_ring.push_back({10.0, y_bot});
        if (i > 0) {
            p.outer_ring.push_back({0.0, y_bot});
            p.outer_ring.push_back({0.0, static_cast<double>(i) * 2.0 - 0.5});
        } else {
            p.outer_ring.push_back({0.0, y_bot});
        }
    }
    p.outer_ring.push_back({0.0, 0.0});
    return p;
}

BooleanPolygon<Point2D> make_vertical_comb(std::size_t teeth) {
    BooleanPolygon<Point2D> p = make_horizontal_comb(teeth);
    for (auto& pt : p.outer_ring) {
        const double x = pt.x;
        pt.x = pt.y;
        pt.y = x;
    }
    return p;
}

void BM_SparseDisjoint(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = make_jittered_star(n, 0.0, 0.0, 50.0, 1);
    const auto b = make_jittered_star(n, 500.0, 0.0, 50.0, 2);

    for (auto _ : state) {
        auto res = polygon_union(a, b);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(2 * n));
}

void BM_GisLinearCrossings(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = make_jittered_star(n, 0.0, 0.0, 100.0, 42);
    const auto b = make_jittered_star(n, 25.0, 10.0, 100.0, 99);

    for (auto _ : state) {
        auto res = polygon_intersection(a, b);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(2 * n));
}

void BM_DenseQuadraticCombGrid(benchmark::State& state) {
    const auto teeth = static_cast<std::size_t>(state.range(0));
    const auto h = make_horizontal_comb(teeth);
    const auto v = make_vertical_comb(teeth);

    for (auto _ : state) {
        auto res = polygon_intersection(h, v);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(teeth * teeth));
}

void BM_AllOpsStar1024(benchmark::State& state) {
    const auto op = static_cast<BooleanOp>(state.range(0));
    const auto a = make_jittered_star(1024, 0.0, 0.0, 100.0, 7);
    const auto b = make_jittered_star(1024, 20.0, 15.0, 100.0, 13);

    for (auto _ : state) {
        auto res = polygon_boolean(a, b, op);
        benchmark::DoNotOptimize(res);
    }
}

} // namespace

BENCHMARK(BM_SparseDisjoint)->RangeMultiplier(4)->Range(64, 4096);
BENCHMARK(BM_GisLinearCrossings)->RangeMultiplier(4)->Range(64, 4096);
BENCHMARK(BM_DenseQuadraticCombGrid)->RangeMultiplier(2)->Range(8, 32);
BENCHMARK(BM_AllOpsStar1024)->DenseRange(0, 3, 1);

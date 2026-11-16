/**
 * @file bench_bentley_ottmann.cpp
 * @brief Google Benchmark suite for Bentley-Ottmann 2D Line Segment Intersection Sweep-Line.
 */

#include "algoat/geometry/bentley_ottmann.hpp"
#include "algoat/geometry/predicates.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

using namespace algoat::geometry;

namespace {

std::vector<LineSegment<double>> generate_sparse_segments(std::size_t n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> pos(0.0, 10000.0);
    std::uniform_real_distribution<double> angle(0.0, 6.283185307179586);
    std::uniform_real_distribution<double> len_dist(1.0, 4.0);

    std::vector<LineSegment<double>> segs;
    segs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = pos(rng);
        const double y = pos(rng);
        const double a = angle(rng);
        const double l = len_dist(rng);
        segs.push_back({{x, y}, {x + l * std::cos(a), y + l * std::sin(a)}});
    }
    return segs;
}

std::vector<IntersectionResult<double>>
brute_force_reference(std::span<const LineSegment<double>> segments) {
    const std::size_t n = segments.size();
    struct Box {
        double min_x, min_y, max_x, max_y;
    };
    std::vector<Box> boxes(n);
    for (std::size_t i = 0; i < n; ++i) {
        boxes[i] = {std::min(segments[i].start[0], segments[i].end[0]),
                    std::min(segments[i].start[1], segments[i].end[1]),
                    std::max(segments[i].start[0], segments[i].end[0]),
                    std::max(segments[i].start[1], segments[i].end[1])};
    }

    std::vector<IntersectionResult<double>> out;
    for (std::size_t i = 0; i < n; ++i) {
        const Point2D p0{segments[i].start[0], segments[i].start[1]};
        const Point2D p1{segments[i].end[0], segments[i].end[1]};
        const Box& bi = boxes[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            const Box& bj = boxes[j];
            if (bi.max_x < bj.min_x || bj.max_x < bi.min_x || bi.max_y < bj.min_y ||
                bj.max_y < bi.min_y) {
                continue;
            }
            const Point2D q0{segments[j].start[0], segments[j].start[1]};
            const Point2D q1{segments[j].end[0], segments[j].end[1]};
            const Orientation o1 = orient2d(p0, p1, q0);
            const Orientation o2 = orient2d(p0, p1, q1);
            const Orientation o3 = orient2d(q0, q1, p0);
            const Orientation o4 = orient2d(q0, q1, p1);
            if (o1 != o2 && o3 != o4) {
                const double dx1 = p1.x - p0.x;
                const double dy1 = p1.y - p0.y;
                const double dx2 = q1.x - q0.x;
                const double dy2 = q1.y - q0.y;
                const double t =
                    ((q0.x - p0.x) * dy2 - (q0.y - p0.y) * dx2) / (dx1 * dy2 - dy1 * dx2);
                out.push_back(IntersectionResult<double>{{p0.x + t * dx1, p0.y + t * dy1}, i, j});
            }
        }
    }
    return out;
}

void BM_BentleyOttmann_Sparse_100k(benchmark::State& state) {
    constexpr std::size_t kN = 100000;
    const auto segs = generate_sparse_segments(kN);
    std::size_t k_count = 0;

    for (auto _ : state) {
        auto res = bentley_ottmann(std::span<const LineSegment<double>>{segs});
        k_count = res.size();
        benchmark::DoNotOptimize(res.data());
    }
    state.counters["N"] = static_cast<double>(kN);
    state.counters["K"] = static_cast<double>(k_count);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));
}

void BM_CountIntersections_Sparse_100k(benchmark::State& state) {
    constexpr std::size_t kN = 100000;
    const auto segs = generate_sparse_segments(kN);
    std::size_t k_count = 0;

    for (auto _ : state) {
        k_count = count_intersections(std::span<const LineSegment<double>>{segs});
        benchmark::DoNotOptimize(k_count);
    }
    state.counters["N"] = static_cast<double>(kN);
    state.counters["K"] = static_cast<double>(k_count);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));
}

void BM_BruteForce_Sparse_100k(benchmark::State& state) {
    constexpr std::size_t kN = 100000;
    const auto segs = generate_sparse_segments(kN);
    std::size_t k_count = 0;

    for (auto _ : state) {
        auto res = brute_force_reference(std::span<const LineSegment<double>>{segs});
        k_count = res.size();
        benchmark::DoNotOptimize(res.data());
    }
    state.counters["N"] = static_cast<double>(kN);
    state.counters["K"] = static_cast<double>(k_count);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));
}

void BM_BentleyOttmann_Scaling(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto segs = generate_sparse_segments(n);
    std::size_t k_count = 0;

    for (auto _ : state) {
        auto res = bentley_ottmann(std::span<const LineSegment<double>>{segs});
        k_count = res.size();
        benchmark::DoNotOptimize(res.data());
    }
    state.counters["K"] = static_cast<double>(k_count);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}

} // namespace

BENCHMARK(BM_BentleyOttmann_Sparse_100k)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CountIntersections_Sparse_100k)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BruteForce_Sparse_100k)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(BM_BentleyOttmann_Scaling)
    ->RangeMultiplier(10)
    ->Range(1000, 100000)
    ->Unit(benchmark::kMillisecond);

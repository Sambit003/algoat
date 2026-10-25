#include "algoat/geometry/kdtree.hpp"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using namespace algoat::geometry;

using Point3D = std::array<double, 3>;

static std::vector<Point3D> generate_random_points(std::size_t count, double min_val = -100.0,
                                                   double max_val = 100.0) {
    std::vector<Point3D> points(count);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (std::size_t i = 0; i < count; ++i) {
        points[i] = {dist(rng), dist(rng), dist(rng)};
    }
    return points;
}

static std::vector<Point3D> generate_queries(std::size_t count, double min_val = -100.0,
                                             double max_val = 100.0) {
    std::vector<Point3D> queries(count);
    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (std::size_t i = 0; i < count; ++i) {
        queries[i] = {dist(rng), dist(rng), dist(rng)};
    }
    return queries;
}

static void BM_BruteForce_KNN_3D(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto points = generate_random_points(n);
    const auto queries = generate_queries(100);
    const std::size_t k = 10;

    std::size_t q_idx = 0;
    for (auto _ : state) {
        const auto& q = queries[q_idx];
        std::vector<std::pair<double, std::size_t>> bf;
        bf.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            double d0 = q[0] - points[i][0];
            double d1 = q[1] - points[i][1];
            double d2 = q[2] - points[i][2];
            bf.emplace_back(d0 * d0 + d1 * d1 + d2 * d2, i);
        }
        std::partial_sort(bf.begin(), bf.begin() + std::min(k, n), bf.end());
        benchmark::DoNotOptimize(bf.data());
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_BruteForce_KNN_3D)->RangeMultiplier(10)->Range(1000, 100000)->Complexity();

static void BM_Algoat_KDTree_KNN_3D(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto points = generate_random_points(n);
    const KDTree<Point3D, 3> tree(points);
    const auto queries = generate_queries(10000);
    const std::size_t k = 10;

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto res = tree.knn_search(queries[q_idx], k);
        benchmark::DoNotOptimize(res.data());
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Algoat_KDTree_KNN_3D)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

static void BM_Algoat_KDTree_Radius_3D(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto points = generate_random_points(n);
    const KDTree<Point3D, 3> tree(points);
    const auto queries = generate_queries(10000);
    const double radius = 5.0;

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto res = tree.radius_search(queries[q_idx], radius);
        benchmark::DoNotOptimize(res.data());
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Algoat_KDTree_Radius_3D)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

static void BM_Algoat_KDTree_Box_3D(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto points = generate_random_points(n);
    const KDTree<Point3D, 3> tree(points);
    const auto queries = generate_queries(10000);

    std::size_t q_idx = 0;
    for (auto _ : state) {
        const auto& q = queries[q_idx];
        Point3D min_c{q[0] - 2.5, q[1] - 2.5, q[2] - 2.5};
        Point3D max_c{q[0] + 2.5, q[1] + 2.5, q[2] + 2.5};
        auto res = tree.box_search(min_c, max_c);
        benchmark::DoNotOptimize(res.data());
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Algoat_KDTree_Box_3D)->RangeMultiplier(10)->Range(1000, 1000000)->Complexity();

static void BM_Algoat_KDTree_Build_3D(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto points = generate_random_points(n);

    for (auto _ : state) {
        KDTree<Point3D, 3> tree(points);
        benchmark::DoNotOptimize(tree.size());
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Algoat_KDTree_Build_3D)->RangeMultiplier(10)->Range(1000, 100000)->Complexity();

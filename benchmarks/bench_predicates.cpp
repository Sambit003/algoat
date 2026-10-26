#include "algoat/geometry/predicates.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <random>
#include <vector>

using namespace algoat::geometry;

struct Query2D {
    Point2D a, b, c;
};

struct Query3D {
    Point3D a, b, c, d;
};

struct InCircleQuery {
    Point2D a, b, c, d;
};

struct InSphereQuery {
    Point3D a, b, c, d, e;
};

static void BM_Orient2D_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    std::vector<Query2D> queries(N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 0; i < N; ++i) {
        queries[i] = {{dist(rng), dist(rng)}, {dist(rng), dist(rng)}, {dist(rng), dist(rng)}};
    }
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = orient2d(q.a, q.b, q.c);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_Orient2D_NonDegenerate);

static void BM_Orient3D_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    std::vector<Query3D> queries(N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 0; i < N; ++i) {
        queries[i] = {{dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)}};
    }
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = orient3d(q.a, q.b, q.c, q.d);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_Orient3D_NonDegenerate);

static void BM_InCircle_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    std::vector<InCircleQuery> queries(N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 0; i < N; ++i) {
        queries[i] = {{dist(rng), dist(rng)},
                      {dist(rng), dist(rng)},
                      {dist(rng), dist(rng)},
                      {dist(rng), dist(rng)}};
    }
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = incircle(q.a, q.b, q.c, q.d);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_InCircle_NonDegenerate);

static void BM_InSphere_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    std::vector<InSphereQuery> queries(N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 0; i < N; ++i) {
        queries[i] = {{dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)}};
    }
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = insphere(q.a, q.b, q.c, q.d, q.e);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_InSphere_NonDegenerate);

static void BM_Orient2D_Collinear(benchmark::State& state) {
    Point2D a{0.5, 0.5};
    Point2D b{12.0, 12.0};
    Point2D c{24.0, 24.0};
    for (auto _ : state) {
        auto res = orient2d(a, b, c);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_Orient2D_Collinear);

static void BM_Orient3D_Coplanar(benchmark::State& state) {
    Point3D a{0.0, 0.0, 0.0};
    Point3D b{1.0, 0.0, 0.0};
    Point3D c{0.0, 1.0, 0.0};
    Point3D d{0.5, 0.5, 0.0};
    for (auto _ : state) {
        auto res = orient3d(a, b, c, d);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_Orient3D_Coplanar);

static void BM_InCircle_Cocircular(benchmark::State& state) {
    Point2D a{1.0, 0.0};
    Point2D b{0.0, 1.0};
    Point2D c{-1.0, 0.0};
    Point2D d{0.0, -1.0};
    for (auto _ : state) {
        auto res = incircle(a, b, c, d);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_InCircle_Cocircular);

static void BM_InSphere_Cospherical(benchmark::State& state) {
    Point3D a{1.0, 0.0, 0.0};
    Point3D b{-1.0, 0.0, 0.0};
    Point3D c{0.0, 1.0, 0.0};
    Point3D d{0.0, -1.0, 0.0};
    Point3D e{0.0, 0.0, 1.0};
    for (auto _ : state) {
        auto res = insphere(a, b, c, d, e);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_InSphere_Cospherical);

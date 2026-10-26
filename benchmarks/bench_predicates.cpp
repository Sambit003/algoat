#include "algoat/geometry/predicates.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <random>
#include <vector>

using namespace algoat::geometry;

// ============================================================================
// Standard C++ Naive Floating-Point Baselines (Unchecked Determinants)
// ============================================================================

[[nodiscard]] static inline Orientation naive_orient2d(const Point2D& pa, const Point2D& pb,
                                                       const Point2D& pc) noexcept {
    double det = (pa.x - pc.x) * (pb.y - pc.y) - (pa.y - pc.y) * (pb.x - pc.x);
    if (det > 0.0)
        return Orientation::CounterClockwise;
    if (det < 0.0)
        return Orientation::Clockwise;
    return Orientation::Collinear;
}

[[nodiscard]] static inline Orientation naive_orient3d(const Point3D& pa, const Point3D& pb,
                                                       const Point3D& pc,
                                                       const Point3D& pd) noexcept {
    double adx = pa.x - pd.x, ady = pa.y - pd.y, adz = pa.z - pd.z;
    double bdx = pb.x - pd.x, bdy = pb.y - pd.y, bdz = pb.z - pd.z;
    double cdx = pc.x - pd.x, cdy = pc.y - pd.y, cdz = pc.z - pd.z;
    double det = adx * (bdy * cdz - bdz * cdy) + bdx * (cdy * adz - cdz * ady) +
                 cdx * (ady * bdz - adz * bdy);
    if (det > 0.0)
        return Orientation::CounterClockwise;
    if (det < 0.0)
        return Orientation::Clockwise;
    return Orientation::Collinear;
}

[[nodiscard]] static inline CircleLocation naive_incircle(const Point2D& pa, const Point2D& pb,
                                                          const Point2D& pc,
                                                          const Point2D& pd) noexcept {
    double adx = pa.x - pd.x, ady = pa.y - pd.y;
    double bdx = pb.x - pd.x, bdy = pb.y - pd.y;
    double cdx = pc.x - pd.x, cdy = pc.y - pd.y;
    double abdet = adx * bdy - bdx * ady;
    double bcdet = bdx * cdy - cdx * bdy;
    double cadet = cdx * ady - adx * cdy;
    double alift = adx * adx + ady * ady;
    double blift = bdx * bdx + bdy * bdy;
    double clift = cdx * cdx + cdy * cdy;
    double det = alift * bcdet + blift * cadet + clift * abdet;
    if (det > 0.0)
        return CircleLocation::Inside;
    if (det < 0.0)
        return CircleLocation::Outside;
    return CircleLocation::Cocirular;
}

[[nodiscard]] static inline CircleLocation naive_insphere(const Point3D& pa, const Point3D& pb,
                                                          const Point3D& pc, const Point3D& pd,
                                                          const Point3D& pe) noexcept {
    double aex = pa.x - pe.x, aey = pa.y - pe.y, aez = pa.z - pe.z;
    double bex = pb.x - pe.x, bey = pb.y - pe.y, bez = pb.z - pe.z;
    double cex = pc.x - pe.x, cey = pc.y - pe.y, cez = pc.z - pe.z;
    double dex = pd.x - pe.x, dey = pd.y - pe.y, dez = pd.z - pe.z;

    double a_lift = aex * aex + aey * aey + aez * aez;
    double b_lift = bex * bex + bey * bey + bez * bez;
    double c_lift = cex * cex + cey * cey + cez * cez;
    double d_lift = dex * dex + dey * dey + dez * dez;

    double bcd = bex * (cey * dez - cez * dey) - bey * (cex * dez - cez * dex) +
                 bez * (cex * dey - cey * dex);
    double acd = aex * (cey * dez - cez * dey) - aey * (cex * dez - cez * dex) +
                 aez * (cex * dey - cey * dex);
    double abd = aex * (bey * dez - bez * dey) - aey * (bex * dez - bez * dex) +
                 aez * (bex * dey - bey * dex);
    double abc = aex * (bey * cez - bez * cey) - aey * (bex * cez - bez * cex) +
                 aez * (bex * cey - bey * cex);

    double det = (a_lift * bcd - b_lift * acd) + (c_lift * abd - d_lift * abc);
    if (det > 0.0)
        return CircleLocation::Inside;
    if (det < 0.0)
        return CircleLocation::Outside;
    return CircleLocation::Cocirular;
}

#if defined(__GNUC__) || defined(__clang__)
// Software Quad Precision (__float128) Baseline for exactness comparison
[[nodiscard]] static inline Orientation quad_orient2d(const Point2D& pa, const Point2D& pb,
                                                      const Point2D& pc) noexcept {
    __float128 ax = pa.x, ay = pa.y;
    __float128 bx = pb.x, by = pb.y;
    __float128 cx = pc.x, cy = pc.y;
    __float128 det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx);
    if (det > 0)
        return Orientation::CounterClockwise;
    if (det < 0)
        return Orientation::Clockwise;
    return Orientation::Collinear;
}
#endif

// ============================================================================
// Benchmark Fixtures & Queries
// ============================================================================

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

static std::vector<Query2D> make_queries_2d(std::size_t n) {
    std::vector<Query2D> queries(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 0; i < n; ++i) {
        queries[i] = {{dist(rng), dist(rng)}, {dist(rng), dist(rng)}, {dist(rng), dist(rng)}};
    }
    return queries;
}

static std::vector<Query3D> make_queries_3d(std::size_t n) {
    std::vector<Query3D> queries(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 0; i < n; ++i) {
        queries[i] = {{dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)}};
    }
    return queries;
}

static std::vector<InCircleQuery> make_queries_incircle(std::size_t n) {
    std::vector<InCircleQuery> queries(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 0; i < n; ++i) {
        queries[i] = {{dist(rng), dist(rng)},
                      {dist(rng), dist(rng)},
                      {dist(rng), dist(rng)},
                      {dist(rng), dist(rng)}};
    }
    return queries;
}

static std::vector<InSphereQuery> make_queries_insphere(std::size_t n) {
    std::vector<InSphereQuery> queries(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 0; i < n; ++i) {
        queries[i] = {{dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)},
                      {dist(rng), dist(rng), dist(rng)}};
    }
    return queries;
}

// ============================================================================
// 1. orient2d: Naive Baseline vs Quad Precision vs Algoat Adaptive
// ============================================================================

static void BM_Orient2D_Baseline_Naive_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    const auto queries = make_queries_2d(N);
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = naive_orient2d(q.a, q.b, q.c);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_Orient2D_Baseline_Naive_NonDegenerate);

#if defined(__GNUC__) || defined(__clang__)
static void BM_Orient2D_Baseline_QuadPrecision_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    const auto queries = make_queries_2d(N);
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = quad_orient2d(q.a, q.b, q.c);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_Orient2D_Baseline_QuadPrecision_NonDegenerate);
#endif

static void BM_Orient2D_Adaptive_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    const auto queries = make_queries_2d(N);
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = orient2d(q.a, q.b, q.c);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_Orient2D_Adaptive_NonDegenerate);

// ============================================================================
// 2. orient3d: Naive Baseline vs Algoat Adaptive
// ============================================================================

static void BM_Orient3D_Baseline_Naive_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    const auto queries = make_queries_3d(N);
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = naive_orient3d(q.a, q.b, q.c, q.d);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_Orient3D_Baseline_Naive_NonDegenerate);

static void BM_Orient3D_Adaptive_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    const auto queries = make_queries_3d(N);
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = orient3d(q.a, q.b, q.c, q.d);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_Orient3D_Adaptive_NonDegenerate);

// ============================================================================
// 3. incircle: Naive Baseline vs Algoat Adaptive
// ============================================================================

static void BM_InCircle_Baseline_Naive_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    const auto queries = make_queries_incircle(N);
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = naive_incircle(q.a, q.b, q.c, q.d);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_InCircle_Baseline_Naive_NonDegenerate);

static void BM_InCircle_Adaptive_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    const auto queries = make_queries_incircle(N);
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = incircle(q.a, q.b, q.c, q.d);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_InCircle_Adaptive_NonDegenerate);

// ============================================================================
// 4. insphere: Naive Baseline vs Algoat Adaptive
// ============================================================================

static void BM_InSphere_Baseline_Naive_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    const auto queries = make_queries_insphere(N);
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = naive_insphere(q.a, q.b, q.c, q.d, q.e);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_InSphere_Baseline_Naive_NonDegenerate);

static void BM_InSphere_Adaptive_NonDegenerate(benchmark::State& state) {
    constexpr std::size_t N = 1024;
    const auto queries = make_queries_insphere(N);
    std::size_t idx = 0;
    for (auto _ : state) {
        const auto& q = queries[idx];
        auto res = insphere(q.a, q.b, q.c, q.d, q.e);
        benchmark::DoNotOptimize(res);
        idx = (idx + 1) & (N - 1);
    }
}
BENCHMARK(BM_InSphere_Adaptive_NonDegenerate);

// ============================================================================
// 5. Degenerate Pathological Cases (Stage B/C/D Exact Escalation)
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
static void BM_Orient2D_Baseline_QuadPrecision_Collinear(benchmark::State& state) {
    Point2D a{0.5, 0.5};
    Point2D b{12.0, 12.0};
    Point2D c{24.0, 24.0};
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        auto res = quad_orient2d(a, b, c);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_Orient2D_Baseline_QuadPrecision_Collinear);
#endif

static void BM_Orient2D_Adaptive_Collinear(benchmark::State& state) {
    Point2D a{0.5, 0.5};
    Point2D b{12.0, 12.0};
    Point2D c{24.0, 24.0};
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        auto res = orient2d(a, b, c);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_Orient2D_Adaptive_Collinear);

static void BM_Orient3D_Adaptive_Coplanar(benchmark::State& state) {
    Point3D a{0.0, 0.0, 0.0};
    Point3D b{1.0, 0.0, 0.0};
    Point3D c{0.0, 1.0, 0.0};
    Point3D d{0.5, 0.5, 0.0};
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        benchmark::DoNotOptimize(d);
        auto res = orient3d(a, b, c, d);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_Orient3D_Adaptive_Coplanar);

static void BM_InCircle_Adaptive_Cocircular(benchmark::State& state) {
    Point2D a{1.0, 0.0};
    Point2D b{0.0, 1.0};
    Point2D c{-1.0, 0.0};
    Point2D d{0.0, -1.0};
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        benchmark::DoNotOptimize(d);
        auto res = incircle(a, b, c, d);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_InCircle_Adaptive_Cocircular);

static void BM_InSphere_Adaptive_Cospherical(benchmark::State& state) {
    Point3D a{1.0, 0.0, 0.0};
    Point3D b{-1.0, 0.0, 0.0};
    Point3D c{0.0, 1.0, 0.0};
    Point3D d{0.0, -1.0, 0.0};
    Point3D e{0.0, 0.0, 1.0};
    for (auto _ : state) {
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        benchmark::DoNotOptimize(d);
        benchmark::DoNotOptimize(e);
        auto res = insphere(a, b, c, d, e);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_InSphere_Adaptive_Cospherical);

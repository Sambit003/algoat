/**
 * @file bench_rstar_tree.cpp
 * @brief Google Benchmark suite for RStarTree (STR bulk-load, dynamic insert,
 *        AVX2 SoA multi-box range queries, k-NN, and spatial join).
 */

#include "algoat/geometry/rstar_tree.hpp"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#if __has_include(<boost/geometry/index/rtree.hpp>)
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#define ALGOAT_HAS_BOOST_GEOMETRY 1
#endif

using namespace algoat::geometry;

namespace {

template <typename T>
std::vector<std::pair<BoundingBox<T, 2>, std::size_t>>
generate_boxes_2d(std::size_t count, T domain = static_cast<T>(1000.0),
                  T max_extent = static_cast<T>(5.0)) {
    std::vector<std::pair<BoundingBox<T, 2>, std::size_t>> items;
    items.reserve(count);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> pos(0.0, static_cast<double>(domain));
    std::uniform_real_distribution<double> ext(0.2, static_cast<double>(max_extent));
    for (std::size_t i = 0; i < count; ++i) {
        const T x = static_cast<T>(pos(rng));
        const T y = static_cast<T>(pos(rng));
        const T w = static_cast<T>(ext(rng));
        const T h = static_cast<T>(ext(rng));
        items.emplace_back(BoundingBox<T, 2>{{x, y}, {x + w, y + h}}, i);
    }
    return items;
}

template <typename T>
std::vector<BoundingBox<T, 2>> generate_queries_2d(std::size_t count,
                                                   T domain = static_cast<T>(1000.0),
                                                   T window_size = static_cast<T>(25.0)) {
    std::vector<BoundingBox<T, 2>> queries;
    queries.reserve(count);
    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> pos(0.0, static_cast<double>(domain - window_size));
    for (std::size_t i = 0; i < count; ++i) {
        const T x = static_cast<T>(pos(rng));
        const T y = static_cast<T>(pos(rng));
        queries.push_back(BoundingBox<T, 2>{{x, y}, {x + window_size, y + window_size}});
    }
    return queries;
}

void BM_Algoat_RStarTree_BulkLoad(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto items = generate_boxes_2d<double>(n);

    for (auto _ : state) {
        RStarTree<std::size_t, double, 2, 16> tree;
        tree.bulk_load(items);
        benchmark::DoNotOptimize(tree.size());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Algoat_RStarTree_BulkLoad)->RangeMultiplier(10)->Range(1000, 100000);

void BM_Algoat_RStarTree_DynamicInsert(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto items = generate_boxes_2d<double>(n);

    for (auto _ : state) {
        RStarTree<std::size_t, double, 2, 16> tree;
        for (const auto& [box, val] : items) {
            tree.insert(box, val);
        }
        benchmark::DoNotOptimize(tree.size());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Algoat_RStarTree_DynamicInsert)->RangeMultiplier(10)->Range(1000, 10000);

void BM_Algoat_RStarTree_MultiBoxRange_Double(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto items = generate_boxes_2d<double>(n);
    const auto queries = generate_queries_2d<double>(1000);

    RStarTree<std::size_t, double, 2, 16> tree;
    tree.bulk_load(items);

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto hits = tree.search_intersecting(queries[q_idx]);
        benchmark::DoNotOptimize(hits.data());
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Algoat_RStarTree_MultiBoxRange_Double)->RangeMultiplier(10)->Range(10000, 100000);

void BM_Algoat_RStarTree_MultiBoxRange_Float(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto items = generate_boxes_2d<float>(n);
    const auto queries = generate_queries_2d<float>(1000);

    RStarTree<std::size_t, float, 2, 16> tree;
    tree.bulk_load(items);

    std::size_t q_idx = 0;
    for (auto _ : state) {
        auto hits = tree.search_intersecting(queries[q_idx]);
        benchmark::DoNotOptimize(hits.data());
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Algoat_RStarTree_MultiBoxRange_Float)->RangeMultiplier(10)->Range(10000, 100000);

void BM_BruteForce_MultiBoxRange(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto items = generate_boxes_2d<double>(n);
    const auto queries = generate_queries_2d<double>(1000);

    std::size_t q_idx = 0;
    for (auto _ : state) {
        const auto& q = queries[q_idx];
        std::vector<std::size_t> hits;
        for (const auto& [box, val] : items) {
            if (box.intersects(q)) {
                hits.push_back(val);
            }
        }
        benchmark::DoNotOptimize(hits.data());
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BruteForce_MultiBoxRange)->RangeMultiplier(10)->Range(10000, 100000);

void BM_Algoat_RStarTree_KnnQuery(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto items = generate_boxes_2d<double>(n);
    RStarTree<std::size_t, double, 2, 16> tree;
    tree.bulk_load(items);

    std::mt19937_64 rng(99);
    std::uniform_real_distribution<double> dist(0.0, 1000.0);
    std::vector<std::array<double, 2>> pts(1000);
    for (auto& pt : pts) {
        pt = {dist(rng), dist(rng)};
    }

    std::size_t idx = 0;
    for (auto _ : state) {
        auto res = tree.knn_search(pts[idx], 10);
        benchmark::DoNotOptimize(res.data());
        idx = (idx + 1) % pts.size();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Algoat_RStarTree_KnnQuery)->RangeMultiplier(10)->Range(10000, 100000);

#if defined(ALGOAT_HAS_BOOST_GEOMETRY)
void BM_BoostGeometry_RStar_MultiBoxRange(benchmark::State& state) {
    namespace bg = boost::geometry;
    namespace bgi = boost::geometry::index;
    using BgPoint = bg::model::point<double, 2, bg::cs::cartesian>;
    using BgBox = bg::model::box<BgPoint>;
    using BgValue = std::pair<BgBox, std::size_t>;

    const auto n = static_cast<std::size_t>(state.range(0));
    const auto items = generate_boxes_2d<double>(n);
    const auto queries = generate_queries_2d<double>(1000);

    std::vector<BgValue> bg_items;
    bg_items.reserve(n);
    for (const auto& [b, v] : items) {
        bg_items.emplace_back(BgBox(BgPoint(b.min_corner[0], b.min_corner[1]),
                                    BgPoint(b.max_corner[0], b.max_corner[1])),
                              v);
    }
    bgi::rtree<BgValue, bgi::rstar<16>> rtree(bg_items.begin(), bg_items.end());

    std::size_t q_idx = 0;
    for (auto _ : state) {
        const auto& q = queries[q_idx];
        BgBox bg_q(BgPoint(q.min_corner[0], q.min_corner[1]),
                   BgPoint(q.max_corner[0], q.max_corner[1]));
        std::vector<std::size_t> hits;
        for (auto it = rtree.qbegin(bgi::intersects(bg_q)); it != rtree.qend(); ++it) {
            hits.push_back(it->second);
        }
        benchmark::DoNotOptimize(hits.data());
        q_idx = (q_idx + 1) % queries.size();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BoostGeometry_RStar_MultiBoxRange)->RangeMultiplier(10)->Range(10000, 100000);
#endif

} // namespace

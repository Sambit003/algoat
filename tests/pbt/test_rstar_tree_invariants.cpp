/**
 * @file test_rstar_tree_invariants.cpp
 * @brief RapidCheck Property-Based Tests for RStarTree invariants.
 */

#include "algoat/geometry/rstar_tree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <utility>
#include <vector>

using namespace algoat::geometry;

namespace {

template <typename T, std::size_t Dim>
BoundingBox<T, Dim> make_box_from_coords(const std::array<int, 2 * Dim>& raw) {
    BoundingBox<T, Dim> b;
    for (std::size_t d = 0; d < Dim; ++d) {
        const T c1 = static_cast<T>(raw[2 * d]) / static_cast<T>(10);
        const T c2 = static_cast<T>(raw[2 * d + 1]) / static_cast<T>(10);
        b.min_corner[d] = std::min(c1, c2);
        b.max_corner[d] = std::max(c1, c2) + static_cast<T>(0.05);
    }
    return b;
}

template <typename T, std::size_t Dim>
double box_mindist(const std::array<T, Dim>& q, const BoundingBox<T, Dim>& b) {
    double sum = 0.0;
    for (std::size_t d = 0; d < Dim; ++d) {
        double gap = 0.0;
        if (q[d] < b.min_corner[d]) {
            gap = static_cast<double>(b.min_corner[d]) - static_cast<double>(q[d]);
        } else if (q[d] > b.max_corner[d]) {
            gap = static_cast<double>(q[d]) - static_cast<double>(b.max_corner[d]);
        }
        sum += gap * gap;
    }
    return std::sqrt(sum);
}

} // namespace

RC_GTEST_PROP(RStarTreePBT, DynamicInsertAndRemoveMatchesBruteForce,
              (const std::vector<std::array<int, 4>>& raw_boxes,
               const std::array<int, 4>& raw_query)) {
    RC_PRE(!raw_boxes.empty());
    const std::size_t n = std::min<std::size_t>(raw_boxes.size(), 150);

    RStarTree<std::size_t, double, 2, 4> tree;
    std::vector<std::pair<BoundingBox<double, 2>, std::size_t>> items;
    items.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        auto b = make_box_from_coords<double, 2>(raw_boxes[i]);
        items.emplace_back(b, i);
        tree.insert(b, i);
    }
    RC_ASSERT(tree.size() == n);

    // Delete every second item to exercise CondenseTree
    std::vector<bool> alive(n, true);
    std::size_t expected_size = n;
    for (std::size_t i = 0; i < n; i += 2) {
        RC_ASSERT(tree.remove(items[i].first, items[i].second));
        alive[i] = false;
        --expected_size;
    }
    RC_ASSERT(tree.size() == expected_size);

    const auto query = make_box_from_coords<double, 2>(raw_query);
    auto actual = tree.search_intersecting(query);
    std::sort(actual.begin(), actual.end());

    std::vector<std::size_t> expected;
    for (std::size_t i = 0; i < n; ++i) {
        if (alive[i] && items[i].first.intersects(query)) {
            expected.push_back(i);
        }
    }
    RC_ASSERT(actual == expected);
}

RC_GTEST_PROP(RStarTreePBT, StrBulkLoadMatchesBruteForceAndKnnInvariants,
              (const std::vector<std::array<int, 4>>& raw_boxes,
               const std::array<int, 4>& raw_query, const std::array<int, 2>& raw_pt)) {
    RC_PRE(!raw_boxes.empty());
    const std::size_t n = std::min<std::size_t>(raw_boxes.size(), 200);

    std::vector<std::pair<BoundingBox<float, 2>, std::size_t>> items;
    items.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        items.emplace_back(make_box_from_coords<float, 2>(raw_boxes[i]), i);
    }

    RStarTree<std::size_t, float, 2, 8> tree;
    tree.bulk_load(items);
    RC_ASSERT(tree.size() == n);

    const auto query = make_box_from_coords<float, 2>(raw_query);
    auto actual = tree.search_intersecting(query);
    std::sort(actual.begin(), actual.end());

    std::vector<std::size_t> expected;
    for (std::size_t i = 0; i < n; ++i) {
        if (items[i].first.intersects(query)) {
            expected.push_back(i);
        }
    }
    RC_ASSERT(actual == expected);

    const std::array<float, 2> qpt{static_cast<float>(raw_pt[0]) / 10.0f,
                                   static_cast<float>(raw_pt[1]) / 10.0f};
    const std::size_t k = std::min<std::size_t>(7, n);
    auto knn = tree.knn_search(qpt, k);
    RC_ASSERT(knn.size() == k);

    std::vector<double> bf_dists;
    bf_dists.reserve(n);
    for (const auto& [b, _] : items) {
        bf_dists.push_back(box_mindist(qpt, b));
    }
    std::sort(bf_dists.begin(), bf_dists.end());

    for (std::size_t i = 0; i < k; ++i) {
        RC_ASSERT(std::abs(knn[i].second - bf_dists[i]) <= 1e-11 * std::max(1.0, bf_dists[i]));
        if (i > 0) {
            RC_ASSERT(knn[i].second >= knn[i - 1].second);
        }
    }
}

RC_GTEST_PROP(RStarTreePBT, SpatialJoinMatchesBruteForce,
              (const std::vector<std::array<int, 4>>& raw_a,
               const std::vector<std::array<int, 4>>& raw_b)) {
    const std::size_t na = std::min<std::size_t>(raw_a.size(), 60);
    const std::size_t nb = std::min<std::size_t>(raw_b.size(), 60);

    std::vector<std::pair<BoundingBox<double, 2>, std::size_t>> items_a;
    std::vector<std::pair<BoundingBox<double, 2>, std::size_t>> items_b;
    for (std::size_t i = 0; i < na; ++i) {
        items_a.emplace_back(make_box_from_coords<double, 2>(raw_a[i]), i);
    }
    for (std::size_t j = 0; j < nb; ++j) {
        items_b.emplace_back(make_box_from_coords<double, 2>(raw_b[j]), j);
    }

    RStarTree<std::size_t, double, 2, 4> tree_a;
    RStarTree<std::size_t, double, 2, 8> tree_b;
    tree_a.bulk_load(items_a);
    tree_b.bulk_load(items_b);

    auto actual = tree_a.spatial_join(tree_b);
    std::sort(actual.begin(), actual.end());

    std::vector<std::pair<std::size_t, std::size_t>> expected;
    for (const auto& [ba, ia] : items_a) {
        for (const auto& [bb, ib] : items_b) {
            if (ba.intersects(bb)) {
                expected.emplace_back(ia, ib);
            }
        }
    }
    std::sort(expected.begin(), expected.end());
    RC_ASSERT(actual == expected);
}

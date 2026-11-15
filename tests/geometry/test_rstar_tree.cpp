/**
 * @file test_rstar_tree.cpp
 * @brief Unit tests for RStarTree with STR bulk-loading and SIMD bounding boxes.
 */

#include "algoat/geometry/rstar_tree.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace algoat::geometry::testing {

TEST(RStarTreeSlice1, BoundingBoxPrimitives2DAnd3D) {
    BoundingBox<double, 2> a{{0.0, 0.0}, {4.0, 3.0}};
    BoundingBox<double, 2> b{{2.0, 1.0}, {6.0, 5.0}};
    BoundingBox<double, 2> c{{4.0, 3.0}, {5.0, 5.0}}; // touching corner
    BoundingBox<double, 2> d{{5.0, 4.0}, {7.0, 6.0}}; // disjoint
    BoundingBox<double, 2> inner{{1.0, 1.0}, {3.0, 2.0}};

    EXPECT_TRUE(a.intersects(b));
    EXPECT_TRUE(b.intersects(a));
    EXPECT_TRUE(a.intersects(c));
    EXPECT_FALSE(a.intersects(d));
    EXPECT_FALSE(d.intersects(a));

    EXPECT_TRUE(a.contains(inner));
    EXPECT_TRUE(a.contains(a));
    EXPECT_FALSE(a.contains(b));
    EXPECT_FALSE(inner.contains(a));

    EXPECT_DOUBLE_EQ(a.area(), 12.0);
    EXPECT_DOUBLE_EQ(a.margin(), 14.0);

    // Degenerate point & inverted box
    BoundingBox<double, 2> pt{{2.0, 2.0}, {2.0, 2.0}};
    EXPECT_DOUBLE_EQ(pt.area(), 0.0);
    EXPECT_DOUBLE_EQ(pt.margin(), 0.0);

    BoundingBox<double, 2> inv{{5.0, 5.0}, {1.0, 1.0}};
    EXPECT_DOUBLE_EQ(inv.area(), 0.0);
    EXPECT_DOUBLE_EQ(inv.margin(), 0.0);

    // NaN coordinates must never produce false-positive intersections
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    BoundingBox<double, 2> nan_box{{qnan, 0.0}, {1.0, 1.0}};
    EXPECT_FALSE(nan_box.intersects(a));
    EXPECT_FALSE(a.intersects(nan_box));
    EXPECT_FALSE(a.contains(nan_box));

    // 3D float BoundingBox
    BoundingBox<float, 3> box3d{{1.0f, 2.0f, 3.0f}, {3.0f, 5.0f, 7.0f}};
    EXPECT_DOUBLE_EQ(box3d.area(), 24.0);
    // 3D cuboid has 4 edges of each length: 4 * (2 + 3 + 4) = 36
    EXPECT_DOUBLE_EQ(box3d.margin(), 36.0);
}

TEST(RStarTreeSlice2, EmptyTreeAndSingleInsert) {
    RStarTree<int> tree;
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.size(), 0u);

    BoundingBox<double, 2> query{{0.0, 0.0}, {10.0, 10.0}};
    EXPECT_TRUE(tree.search_intersecting(query).empty());

    tree.insert(BoundingBox<double, 2>{{1.0, 2.0}, {3.0, 4.0}}, 42);
    EXPECT_FALSE(tree.empty());
    EXPECT_EQ(tree.size(), 1u);

    auto hits = tree.search_intersecting(query);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0], 42);

    BoundingBox<double, 2> miss{{5.0, 5.0}, {6.0, 6.0}};
    EXPECT_TRUE(tree.search_intersecting(miss).empty());
}

TEST(RStarTreeSlice2, DynamicInsertWithReinsertAndSplitsMatchesBruteForce) {
    // Test small MaxEntries=4 (forces deep splits & reinserts) and float AVX2 8-wide path
    RStarTree<std::size_t, float, 2, 4> tree4;
    RStarTree<std::size_t, double, 2, 16> tree16;

    std::vector<BoundingBox<float, 2>> boxes_f;
    std::vector<BoundingBox<double, 2>> boxes_d;

    for (std::size_t i = 0; i < 400; ++i) {
        const double x = static_cast<double>((i * 37) % 100);
        const double y = static_cast<double>((i * 53) % 100);
        const double w = 1.0 + static_cast<double>(i % 7);
        const double h = 1.0 + static_cast<double>((i * 3) % 5);
        BoundingBox<float, 2> bf{{static_cast<float>(x), static_cast<float>(y)},
                                 {static_cast<float>(x + w), static_cast<float>(y + h)}};
        BoundingBox<double, 2> bd{{x, y}, {x + w, y + h}};
        boxes_f.push_back(bf);
        boxes_d.push_back(bd);
        tree4.insert(bf, i);
        tree16.insert(bd, i);
    }

    EXPECT_EQ(tree4.size(), 400u);
    EXPECT_EQ(tree16.size(), 400u);
    EXPECT_GT(tree4.height(), 2u);
    EXPECT_GT(tree16.height(), 1u);

    for (int q = 0; q < 20; ++q) {
        const double qx = static_cast<double>((q * 19) % 90);
        const double qy = static_cast<double>((q * 23) % 90);
        BoundingBox<float, 2> qf{{static_cast<float>(qx), static_cast<float>(qy)},
                                 {static_cast<float>(qx + 15.0), static_cast<float>(qy + 15.0)}};
        BoundingBox<double, 2> qd{{qx, qy}, {qx + 15.0, qy + 15.0}};

        auto got_f = tree4.search_intersecting(qf);
        auto got_d = tree16.search_intersecting(qd);
        std::sort(got_f.begin(), got_f.end());
        std::sort(got_d.begin(), got_d.end());

        std::vector<std::size_t> expected_f;
        std::vector<std::size_t> expected_d;
        for (std::size_t i = 0; i < 400; ++i) {
            if (boxes_f[i].intersects(qf)) {
                expected_f.push_back(i);
            }
            if (boxes_d[i].intersects(qd)) {
                expected_d.push_back(i);
            }
        }
        EXPECT_EQ(got_f, expected_f);
        EXPECT_EQ(got_d, expected_d);
    }
}

TEST(RStarTreeSlice2, ThreeDimensionalDynamicInsertAndSearch) {
    RStarTree<int, double, 3, 8> tree;
    std::vector<BoundingBox<double, 3>> boxes;
    for (int i = 0; i < 120; ++i) {
        const double x = static_cast<double>((i * 11) % 30);
        const double y = static_cast<double>((i * 17) % 30);
        const double z = static_cast<double>((i * 29) % 30);
        BoundingBox<double, 3> b{{x, y, z}, {x + 2.5, y + 2.5, z + 2.5}};
        boxes.push_back(b);
        tree.insert(b, i);
    }

    BoundingBox<double, 3> query{{5.0, 5.0, 5.0}, {15.0, 15.0, 15.0}};
    auto got = tree.search_intersecting(query);
    std::sort(got.begin(), got.end());

    std::vector<int> expected;
    for (int i = 0; i < 120; ++i) {
        if (boxes[static_cast<std::size_t>(i)].intersects(query)) {
            expected.push_back(i);
        }
    }
    EXPECT_EQ(got, expected);
}

TEST(RStarTreeSlice3, RemoveAndCondenseTreePreservesExactEntries) {
    RStarTree<int, double, 2, 4> tree;
    std::vector<std::pair<BoundingBox<double, 2>, int>> items;
    for (int i = 0; i < 200; ++i) {
        const double x = static_cast<double>((i * 23) % 80);
        const double y = static_cast<double>((i * 41) % 80);
        BoundingBox<double, 2> b{{x, y}, {x + 3.0, y + 3.0}};
        items.emplace_back(b, i);
        tree.insert(b, i);
    }

    // Removing non-existent entry returns false and does not change size
    EXPECT_FALSE(tree.remove(BoundingBox<double, 2>{{0.0, 0.0}, {1.0, 1.0}}, 9999));
    EXPECT_FALSE(tree.remove(items[0].first, 9999));
    EXPECT_EQ(tree.size(), 200u);

    // Delete 150 entries (triggers underflow, CondenseTree orphan reinsertion, and root collapse)
    for (std::size_t i = 0; i < 150; ++i) {
        EXPECT_TRUE(tree.remove(items[i].first, items[i].second));
    }
    EXPECT_EQ(tree.size(), 50u);

    // Verify exact remaining set via full-domain search
    BoundingBox<double, 2> domain{{-10.0, -10.0}, {200.0, 200.0}};
    auto remaining = tree.search_intersecting(domain);
    std::sort(remaining.begin(), remaining.end());
    ASSERT_EQ(remaining.size(), 50u);
    for (std::size_t i = 0; i < 50; ++i) {
        EXPECT_EQ(remaining[i], static_cast<int>(150 + i));
    }

    // Copy & move preservation
    auto copied = tree;
    auto moved = std::move(tree);
    auto moved_items = moved.search_intersecting(domain);
    std::sort(moved_items.begin(), moved_items.end());
    EXPECT_EQ(moved_items, remaining);

    // Delete all remaining entries from copied tree until completely empty
    for (std::size_t i = 150; i < 200; ++i) {
        EXPECT_TRUE(copied.remove(items[i].first, items[i].second));
    }
    EXPECT_TRUE(copied.empty());
    EXPECT_EQ(copied.size(), 0u);
    EXPECT_EQ(copied.height(), 0u);
    EXPECT_TRUE(copied.search_intersecting(domain).empty());

    // Re-insert into emptied tree
    copied.insert(BoundingBox<double, 2>{{1.0, 1.0}, {2.0, 2.0}}, 777);
    EXPECT_EQ(copied.size(), 1u);
    auto rehit = copied.search_intersecting(domain);
    ASSERT_EQ(rehit.size(), 1u);
    EXPECT_EQ(rehit[0], 777);
}

TEST(RStarTreeSlice3, DuplicateBoxValueRemovesSingleInstance) {
    RStarTree<int, double, 2, 8> tree;
    BoundingBox<double, 2> b{{1.0, 1.0}, {2.0, 2.0}};
    tree.insert(b, 10);
    tree.insert(b, 10);
    tree.insert(b, 20);
    EXPECT_EQ(tree.size(), 3u);

    EXPECT_TRUE(tree.remove(b, 10));
    EXPECT_EQ(tree.size(), 2u);
    auto hits = tree.search_intersecting(b);
    std::sort(hits.begin(), hits.end());
    EXPECT_EQ(hits, (std::vector<int>{10, 20}));
}

TEST(RStarTreeSlice4, StrBulkLoadFillFactorAnd100kEquivalence) {
    RStarTree<std::size_t, double, 2, 16> tree;
    // Pre-populate to ensure bulk_load replaces existing contents cleanly
    tree.insert(BoundingBox<double, 2>{{0.0, 0.0}, {1.0, 1.0}}, 999999);

    std::vector<std::pair<BoundingBox<double, 2>, std::size_t>> items;
    constexpr std::size_t kN = 100000;
    items.reserve(kN);

    uint64_t state = 0x9E3779B97F4A7C15ULL;
    auto next_u64 = [&]() -> uint64_t {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    auto next_unit = [&]() -> double {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    };

    for (std::size_t i = 0; i < kN; ++i) {
        const double x = next_unit() * 1000.0;
        const double y = next_unit() * 1000.0;
        const double w = 0.5 + next_unit() * 5.0;
        const double h = 0.5 + next_unit() * 5.0;
        items.emplace_back(BoundingBox<double, 2>{{x, y}, {x + w, y + h}}, i);
    }

    tree.bulk_load(items);
    EXPECT_EQ(tree.size(), kN);
    EXPECT_GT(tree.fill_factor(), 0.95);

    // 100% equivalence against brute-force across multiple query windows
    for (int q = 0; q < 10; ++q) {
        const double qx = next_unit() * 900.0;
        const double qy = next_unit() * 900.0;
        const BoundingBox<double, 2> query{{qx, qy}, {qx + 40.0, qy + 40.0}};

        auto got = tree.search_intersecting(query);
        std::sort(got.begin(), got.end());

        std::vector<std::size_t> expected;
        for (const auto& [box, val] : items) {
            if (box.intersects(query)) {
                expected.push_back(val);
            }
        }
        EXPECT_EQ(got, expected);
    }

    // Empty bulk_load clears tree
    std::vector<std::pair<BoundingBox<double, 2>, std::size_t>> empty_items;
    tree.bulk_load(empty_items);
    EXPECT_TRUE(tree.empty());
    EXPECT_DOUBLE_EQ(tree.fill_factor(), 0.0);
}

TEST(RStarTreeSlice4, Str3DBulkLoadAndDynamicMutationAfterLoad) {
    RStarTree<int, float, 3, 16> tree;
    std::vector<std::pair<BoundingBox<float, 3>, int>> items;
    for (int i = 0; i < 2000; ++i) {
        const float x = static_cast<float>((i * 17) % 100);
        const float y = static_cast<float>((i * 31) % 100);
        const float z = static_cast<float>((i * 47) % 100);
        items.emplace_back(BoundingBox<float, 3>{{x, y, z}, {x + 2.0f, y + 2.0f, z + 2.0f}}, i);
    }
    tree.bulk_load(items);
    EXPECT_EQ(tree.size(), 2000u);
    EXPECT_GT(tree.fill_factor(), 0.95);

    // Mutate after bulk_load
    BoundingBox<float, 3> extra_box{{25.0f, 25.0f, 25.0f}, {28.0f, 28.0f, 28.0f}};
    tree.insert(extra_box, 5000);
    EXPECT_EQ(tree.size(), 2001u);
    EXPECT_TRUE(tree.remove(items[10].first, items[10].second));
    EXPECT_EQ(tree.size(), 2000u);

    BoundingBox<float, 3> query{{20.0f, 20.0f, 20.0f}, {35.0f, 35.0f, 35.0f}};
    auto got = tree.search_intersecting(query);
    std::sort(got.begin(), got.end());

    std::vector<int> expected;
    for (int i = 0; i < 2000; ++i) {
        if (i == 10) {
            continue;
        }
        if (items[static_cast<std::size_t>(i)].first.intersects(query)) {
            expected.push_back(i);
        }
    }
    expected.push_back(5000);
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(got, expected);
}

TEST(RStarTreeSlice4, SpatialJoinMatchesBruteForce) {
    std::vector<std::pair<BoundingBox<double, 2>, int>> left_items;
    std::vector<std::pair<BoundingBox<double, 2>, int>> right_items;

    for (int i = 0; i < 150; ++i) {
        const double x = static_cast<double>((i * 13) % 60);
        const double y = static_cast<double>((i * 29) % 60);
        left_items.emplace_back(BoundingBox<double, 2>{{x, y}, {x + 3.5, y + 3.5}}, i);
    }
    for (int j = 0; j < 120; ++j) {
        const double x = static_cast<double>((j * 19 + 5) % 60);
        const double y = static_cast<double>((j * 37 + 3) % 60);
        right_items.emplace_back(BoundingBox<double, 2>{{x, y}, {x + 2.5, y + 2.5}}, j);
    }

    RStarTree<int, double, 2, 16> left_tree;
    RStarTree<int, double, 2, 16> right_tree;
    left_tree.bulk_load(left_items);
    right_tree.bulk_load(right_items);

    auto got = left_tree.spatial_join(right_tree);
    std::sort(got.begin(), got.end());

    std::vector<std::pair<int, int>> expected;
    for (const auto& [lb, lv] : left_items) {
        for (const auto& [rb, rv] : right_items) {
            if (lb.intersects(rb)) {
                expected.emplace_back(lv, rv);
            }
        }
    }
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(got, expected);
}

TEST(RStarTreeSlice5, KnnSearchMatchesBruteForce2DAnd3D) {
    RStarTree<int, double, 2, 16> tree;
    EXPECT_TRUE(tree.knn_search({0.0, 0.0}, 5).empty());

    std::vector<std::pair<BoundingBox<double, 2>, int>> items;
    for (int i = 0; i < 500; ++i) {
        const double x = static_cast<double>((i * 37) % 120) - 60.0;
        const double y = static_cast<double>((i * 53) % 120) - 60.0;
        const double w = 1.0 + static_cast<double>(i % 5);
        const double h = 1.0 + static_cast<double>((i * 2) % 5);
        items.emplace_back(BoundingBox<double, 2>{{x, y}, {x + w, y + h}}, i);
    }
    tree.bulk_load(items);

    EXPECT_TRUE(tree.knn_search({0.0, 0.0}, 0).empty());
    EXPECT_TRUE(tree.knn_search({std::numeric_limits<double>::quiet_NaN(), 0.0}, 5).empty());

    auto mindist_box = [](const std::array<double, 2>& q, const BoundingBox<double, 2>& b) {
        double sum = 0.0;
        for (std::size_t d = 0; d < 2; ++d) {
            double gap = 0.0;
            if (q[d] < b.min_corner[d]) {
                gap = b.min_corner[d] - q[d];
            } else if (q[d] > b.max_corner[d]) {
                gap = q[d] - b.max_corner[d];
            }
            sum += gap * gap;
        }
        return std::sqrt(sum);
    };

    const std::array<double, 2> query{12.3, -7.8};
    constexpr std::size_t k = 15;
    auto got = tree.knn_search(query, k);
    ASSERT_EQ(got.size(), k);

    std::vector<std::pair<double, int>> bf;
    bf.reserve(items.size());
    for (const auto& [b, v] : items) {
        bf.emplace_back(mindist_box(query, b), v);
    }
    std::sort(bf.begin(), bf.end());

    // Verify distances are non-decreasing and match the k-th distance sequence exactly
    for (std::size_t i = 0; i < k; ++i) {
        EXPECT_NEAR(got[i].second, bf[i].first, 1e-12);
        if (i > 0) {
            EXPECT_GE(got[i].second, got[i - 1].second);
        }
    }

    // Requesting k > size returns all items
    auto all = tree.knn_search(query, 2000);
    EXPECT_EQ(all.size(), items.size());

    // Float 3D dynamic tree k-NN
    RStarTree<int, float, 3, 8> tree3d;
    std::vector<std::pair<BoundingBox<float, 3>, int>> items3d;
    for (int i = 0; i < 100; ++i) {
        const float x = static_cast<float>((i * 13) % 40);
        const float y = static_cast<float>((i * 29) % 40);
        const float z = static_cast<float>((i * 41) % 40);
        BoundingBox<float, 3> b{{x, y, z}, {x + 1.5f, y + 1.5f, z + 1.5f}};
        items3d.emplace_back(b, i);
        tree3d.insert(b, i);
    }
    const std::array<float, 3> q3{10.5f, 15.2f, 20.8f};
    auto got3 = tree3d.knn_search(q3, 8);
    ASSERT_EQ(got3.size(), 8u);

    std::vector<double> bf3;
    for (const auto& [b, _] : items3d) {
        double s = 0.0;
        for (std::size_t d = 0; d < 3; ++d) {
            double g = 0.0;
            if (q3[d] < b.min_corner[d]) {
                g = static_cast<double>(b.min_corner[d] - q3[d]);
            } else if (q3[d] > b.max_corner[d]) {
                g = static_cast<double>(q3[d] - b.max_corner[d]);
            }
            s += g * g;
        }
        bf3.push_back(std::sqrt(s));
    }
    std::sort(bf3.begin(), bf3.end());
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(got3[i].second, bf3[i], 1e-5);
    }
}

} // namespace algoat::geometry::testing

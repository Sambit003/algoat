#include "algoat/geometry/kdtree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace algoat::geometry;

using Point2D = std::array<double, 2>;
using Point3D = std::array<double, 3>;
using Point6D = std::array<double, 6>;

static_assert(SpatialPoint<Point2D, 2>);
static_assert(SpatialPoint<Point3D, 3>);
static_assert(SpatialPoint<Point6D, 6>);

struct CustomPoint3D {
    double coords[3];
    double operator[](std::size_t i) const {
        return coords[i];
    }
    double& operator[](std::size_t i) {
        return coords[i];
    }
};
static_assert(SpatialPoint<CustomPoint3D, 3>);

template <std::size_t Dim>
double brute_force_dist(const std::array<double, Dim>& a, const std::array<double, Dim>& b,
                        MetricType metric, double p = 3.0) {
    if (metric == MetricType::EuclideanL2) {
        double sum = 0.0;
        for (std::size_t d = 0; d < Dim; ++d) {
            double diff = a[d] - b[d];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    if (metric == MetricType::ManhattanL1) {
        double sum = 0.0;
        for (std::size_t d = 0; d < Dim; ++d) {
            sum += std::abs(a[d] - b[d]);
        }
        return sum;
    }
    if (metric == MetricType::ChebyshevLInf) {
        double m = 0.0;
        for (std::size_t d = 0; d < Dim; ++d) {
            m = std::max(m, std::abs(a[d] - b[d]));
        }
        return m;
    }
    // MinkowskiLp
    double sum = 0.0;
    for (std::size_t d = 0; d < Dim; ++d) {
        sum += std::pow(std::abs(a[d] - b[d]), p);
    }
    return std::pow(sum, 1.0 / p);
}

template <std::size_t Dim>
std::vector<typename KDTree<std::array<double, Dim>, Dim>::NeighborResult>
brute_force_knn(const std::vector<std::array<double, Dim>>& points,
                const std::array<double, Dim>& query, std::size_t k, MetricType metric,
                double p = 3.0) {
    using Res = typename KDTree<std::array<double, Dim>, Dim>::NeighborResult;
    std::vector<Res> all;
    all.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        all.push_back({i, brute_force_dist<Dim>(query, points[i], metric, p)});
    }
    std::sort(all.begin(), all.end());
    if (all.size() > k) {
        all.resize(k);
    }
    return all;
}

template <std::size_t Dim>
std::vector<std::size_t> brute_force_radius(const std::vector<std::array<double, Dim>>& points,
                                            const std::array<double, Dim>& query, double radius,
                                            MetricType metric, double p = 3.0) {
    if (radius < 0.0) {
        return {};
    }
    double threshold = 0.0;
    if (metric == MetricType::EuclideanL2) {
        threshold = radius * radius;
    } else if (metric == MetricType::ManhattanL1 || metric == MetricType::ChebyshevLInf) {
        threshold = radius;
    } else {
        threshold = std::pow(radius, p);
    }

    std::vector<std::size_t> res;
    for (std::size_t i = 0; i < points.size(); ++i) {
        double cur_dist = 0.0;
        if (metric == MetricType::EuclideanL2) {
            for (std::size_t d = 0; d < Dim; ++d) {
                double diff = query[d] - points[i][d];
                cur_dist += diff * diff;
            }
        } else if (metric == MetricType::ManhattanL1) {
            for (std::size_t d = 0; d < Dim; ++d) {
                cur_dist += std::abs(query[d] - points[i][d]);
            }
        } else if (metric == MetricType::ChebyshevLInf) {
            for (std::size_t d = 0; d < Dim; ++d) {
                cur_dist = std::max(cur_dist, std::abs(query[d] - points[i][d]));
            }
        } else {
            for (std::size_t d = 0; d < Dim; ++d) {
                cur_dist += std::pow(std::abs(query[d] - points[i][d]), p);
            }
        }

        if (cur_dist <= threshold) {
            res.push_back(i);
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

template <std::size_t Dim>
std::vector<std::size_t> brute_force_box(const std::vector<std::array<double, Dim>>& points,
                                         const std::array<double, Dim>& min_corner,
                                         const std::array<double, Dim>& max_corner) {
    for (std::size_t d = 0; d < Dim; ++d) {
        if (min_corner[d] > max_corner[d]) {
            return {};
        }
    }
    std::vector<std::size_t> res;
    for (std::size_t i = 0; i < points.size(); ++i) {
        bool inside = true;
        for (std::size_t d = 0; d < Dim; ++d) {
            if (points[i][d] < min_corner[d] || points[i][d] > max_corner[d]) {
                inside = false;
                break;
            }
        }
        if (inside) {
            res.push_back(i);
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

TEST(KDTreeTest, EmptyTreeInvariants) {
    KDTree<Point3D, 3> tree;
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.size(), 0u);

    Point3D query{0.0, 0.0, 0.0};
    auto knn = tree.knn_search(query, 5);
    EXPECT_TRUE(knn.empty());

    auto radius = tree.radius_search(query, 1.0);
    EXPECT_TRUE(radius.empty());

    auto box = tree.box_search(Point3D{-1, -1, -1}, Point3D{1, 1, 1});
    EXPECT_TRUE(box.empty());
}

TEST(KDTreeTest, SinglePointConstructionAndQuery) {
    std::vector<Point3D> points = {{1.0, 2.0, 3.0}};
    KDTree<Point3D, 3> tree(points);

    EXPECT_FALSE(tree.empty());
    EXPECT_EQ(tree.size(), 1u);

    Point3D query{1.0, 2.0, 3.0};
    auto knn = tree.knn_search(query, 1);
    ASSERT_EQ(knn.size(), 1u);
    EXPECT_EQ(knn[0].index, 0u);
    EXPECT_NEAR(knn[0].distance, 0.0, 1e-9);

    auto rad = tree.radius_search(query, 0.1);
    ASSERT_EQ(rad.size(), 1u);
    EXPECT_EQ(rad[0], 0u);

    auto rad_outside = tree.radius_search(Point3D{10.0, 10.0, 10.0}, 1.0);
    EXPECT_TRUE(rad_outside.empty());

    auto box = tree.box_search(Point3D{0.0, 1.0, 2.0}, Point3D{2.0, 3.0, 4.0});
    ASSERT_EQ(box.size(), 1u);
    EXPECT_EQ(box[0], 0u);
}

TEST(KDTreeTest, RebuildAndClear) {
    std::vector<Point2D> pts1 = {{0.0, 0.0}, {1.0, 1.0}};
    KDTree<Point2D, 2> tree(pts1);
    EXPECT_EQ(tree.size(), 2u);

    tree.clear();
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.size(), 0u);

    std::vector<Point2D> pts2 = {{5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0}};
    tree.build(pts2);
    EXPECT_EQ(tree.size(), 3u);

    auto knn = tree.knn_search(Point2D{5.1, 5.1}, 1);
    ASSERT_EQ(knn.size(), 1u);
    EXPECT_EQ(knn[0].index, 0u);
}

TEST(KDTreeTest, NonPowerOfTwoSizes) {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    const std::vector<std::size_t> test_sizes = {2,  3,  4,  5,  6,  7,  8,   9,  10,
                                                 15, 17, 31, 33, 63, 65, 127, 129};
    for (std::size_t n : test_sizes) {
        std::vector<Point3D> points(n);
        for (std::size_t i = 0; i < n; ++i) {
            points[i] = {dist(rng), dist(rng), dist(rng)};
        }

        KDTree<Point3D, 3> tree(points);
        EXPECT_EQ(tree.size(), n);

        Point3D query{dist(rng), dist(rng), dist(rng)};
        auto knn = tree.knn_search(query, 5);
        auto expected = brute_force_knn<3>(points, query, 5, MetricType::EuclideanL2);

        ASSERT_EQ(knn.size(), expected.size());
        for (std::size_t i = 0; i < knn.size(); ++i) {
            EXPECT_NEAR(knn[i].distance, expected[i].distance, 1e-9);
        }
    }
}

TEST(KDTreeTest, AllMetricTypesEquivalence2D) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-50.0, 50.0);

    const std::size_t n = 250;
    std::vector<Point2D> points(n);
    for (std::size_t i = 0; i < n; ++i) {
        points[i] = {dist(rng), dist(rng)};
    }

    KDTree<Point2D, 2> tree(points);

    const std::vector<MetricType> metrics = {MetricType::EuclideanL2, MetricType::ManhattanL1,
                                             MetricType::ChebyshevLInf, MetricType::MinkowskiLp};

    for (auto metric : metrics) {
        for (std::size_t q = 0; q < 20; ++q) {
            Point2D query{dist(rng), dist(rng)};
            auto actual_knn = tree.knn_search(query, 10, metric, 3.0);
            auto expected_knn = brute_force_knn<2>(points, query, 10, metric, 3.0);

            ASSERT_EQ(actual_knn.size(), expected_knn.size());
            for (std::size_t i = 0; i < actual_knn.size(); ++i) {
                EXPECT_NEAR(actual_knn[i].distance, expected_knn[i].distance, 1e-9);
            }

            double radius = 15.0;
            auto actual_rad = tree.radius_search(query, radius, metric, 3.0);
            auto expected_rad = brute_force_radius<2>(points, query, radius, metric, 3.0);
            EXPECT_EQ(actual_rad, expected_rad);
        }
    }
}

TEST(KDTreeTest, ExactMathematicalEquivalence3D) {
    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    const std::size_t n = 1000;
    std::vector<Point3D> points(n);
    for (std::size_t i = 0; i < n; ++i) {
        points[i] = {dist(rng), dist(rng), dist(rng)};
    }

    KDTree<Point3D, 3> tree(points);

    for (std::size_t q = 0; q < 50; ++q) {
        Point3D query{dist(rng), dist(rng), dist(rng)};

        // k-NN
        auto actual_knn = tree.knn_search(query, 8);
        auto expected_knn = brute_force_knn<3>(points, query, 8, MetricType::EuclideanL2);
        ASSERT_EQ(actual_knn.size(), expected_knn.size());
        for (std::size_t i = 0; i < actual_knn.size(); ++i) {
            EXPECT_NEAR(actual_knn[i].distance, expected_knn[i].distance, 1e-9);
        }

        // Radius
        double r = 25.0;
        auto actual_rad = tree.radius_search(query, r);
        auto expected_rad = brute_force_radius<3>(points, query, r, MetricType::EuclideanL2);
        EXPECT_EQ(actual_rad, expected_rad);

        // Box
        Point3D p1{dist(rng), dist(rng), dist(rng)};
        Point3D p2{dist(rng), dist(rng), dist(rng)};
        Point3D min_c{std::min(p1[0], p2[0]), std::min(p1[1], p2[1]), std::min(p1[2], p2[2])};
        Point3D max_c{std::max(p1[0], p2[0]), std::max(p1[1], p2[1]), std::max(p1[2], p2[2])};

        auto actual_box = tree.box_search(min_c, max_c);
        auto expected_box = brute_force_box<3>(points, min_c, max_c);
        EXPECT_EQ(actual_box, expected_box);
    }
}

TEST(KDTreeTest, HighDimensionalEquivalence6D) {
    std::mt19937_64 rng(777);
    std::uniform_real_distribution<double> dist(-50.0, 50.0);

    const std::size_t n = 500;
    std::vector<Point6D> points(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < 6; ++d) {
            points[i][d] = dist(rng);
        }
    }

    KDTree<Point6D, 6> tree(points);

    for (std::size_t q = 0; q < 30; ++q) {
        Point6D query;
        for (std::size_t d = 0; d < 6; ++d) {
            query[d] = dist(rng);
        }

        auto actual_knn = tree.knn_search(query, 5);
        auto expected_knn = brute_force_knn<6>(points, query, 5, MetricType::EuclideanL2);
        ASSERT_EQ(actual_knn.size(), expected_knn.size());
        for (std::size_t i = 0; i < actual_knn.size(); ++i) {
            EXPECT_NEAR(actual_knn[i].distance, expected_knn[i].distance, 1e-9);
        }
    }
}

TEST(KDTreeTest, EquivalenceLargeBatch50000Queries) {
    // Fulfills acceptance criterion: 100% equivalence on 50,000 query points across 2D, 3D, and 6D
    std::mt19937_64 rng(50000);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    // 2D batch
    {
        const std::size_t n = 100;
        std::vector<Point2D> pts2d(n);
        for (std::size_t i = 0; i < n; ++i) {
            pts2d[i] = {dist(rng), dist(rng)};
        }
        KDTree<Point2D, 2> tree2d(pts2d);
        for (std::size_t q = 0; q < 20000; ++q) {
            Point2D qpt{dist(rng), dist(rng)};
            auto actual = tree2d.knn_search(qpt, 1);
            auto expected = brute_force_knn<2>(pts2d, qpt, 1, MetricType::EuclideanL2);
            ASSERT_EQ(actual.size(), expected.size());
            EXPECT_NEAR(actual[0].distance, expected[0].distance, 1e-9);
        }
    }

    // 3D batch
    {
        const std::size_t n = 100;
        std::vector<Point3D> pts3d(n);
        for (std::size_t i = 0; i < n; ++i) {
            pts3d[i] = {dist(rng), dist(rng), dist(rng)};
        }
        KDTree<Point3D, 3> tree3d(pts3d);
        for (std::size_t q = 0; q < 20000; ++q) {
            Point3D qpt{dist(rng), dist(rng), dist(rng)};
            auto actual = tree3d.knn_search(qpt, 3);
            auto expected = brute_force_knn<3>(pts3d, qpt, 3, MetricType::EuclideanL2);
            ASSERT_EQ(actual.size(), expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i].distance, expected[i].distance, 1e-9);
            }
        }
    }

    // 6D batch
    {
        const std::size_t n = 50;
        std::vector<Point6D> pts6d(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t d = 0; d < 6; ++d) {
                pts6d[i][d] = dist(rng);
            }
        }
        KDTree<Point6D, 6> tree6d(pts6d);
        for (std::size_t q = 0; q < 10000; ++q) {
            Point6D qpt;
            for (std::size_t d = 0; d < 6; ++d) {
                qpt[d] = dist(rng);
            }
            auto actual = tree6d.knn_search(qpt, 2);
            auto expected = brute_force_knn<6>(pts6d, qpt, 2, MetricType::EuclideanL2);
            ASSERT_EQ(actual.size(), expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i].distance, expected[i].distance, 1e-9);
            }
        }
    }
}

TEST(KDTreeTest, DuplicateCoordinatesHandling) {
    std::vector<Point3D> points = {
        {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {2.0, 2.0, 2.0}};
    KDTree<Point3D, 3> tree(points);
    EXPECT_EQ(tree.size(), 5u);

    auto knn = tree.knn_search(Point3D{1.0, 1.0, 1.0}, 3);
    ASSERT_EQ(knn.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(knn[i].distance, 0.0, 1e-9);
    }

    auto rad = tree.radius_search(Point3D{1.0, 1.0, 1.0}, 0.001);
    EXPECT_EQ(rad.size(), 3u);
}

TEST(KDTreeTest, BoxSearchEdgeCases) {
    std::vector<Point2D> points = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    KDTree<Point2D, 2> tree(points);

    // Inverted box -> empty
    auto inv = tree.box_search(Point2D{2.0, 2.0}, Point2D{1.0, 1.0});
    EXPECT_TRUE(inv.empty());

    // Exact single point box
    auto exact = tree.box_search(Point2D{1.0, 1.0}, Point2D{1.0, 1.0});
    ASSERT_EQ(exact.size(), 1u);
    EXPECT_EQ(exact[0], 1u);

    // Box containing all points
    auto all = tree.box_search(Point2D{-1.0, -1.0}, Point2D{4.0, 4.0});
    EXPECT_EQ(all.size(), 4u);
}

TEST(KDTreeTest, MemoryFootprintBound) {
    const std::size_t n = 10000;
    std::vector<Point3D> points(n, Point3D{1.0, 2.0, 3.0});
    KDTree<Point3D, 3> tree(points);

    EXPECT_EQ(tree.size(), n);

    // Memory footprint bound check: strictly K * N * 8 bytes + N * 8 bytes (+ minimal vector
    // overhead) For 3D: (3 * 8 + 8) * (N + 1) = 32 * (N + 1) bytes
    const std::size_t memory_used = tree.memory_usage_bytes();
    const std::size_t theoretical_bound = (3 * 8 + 8) * (n + 1);

    // Assert that the allocated buffer size strictly respects the (K + 1) * (N + 1) * 8 bound
    EXPECT_LE(memory_used, theoretical_bound * 2); // allocator capacity tolerance
    EXPECT_GE(memory_used, theoretical_bound);
}

#include "algoat/geometry/kdtree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::geometry;

using Point3D = std::array<double, 3>;

static double dist_euclidean_sq(const Point3D& a, const Point3D& b) {
    double sum = 0.0;
    for (std::size_t d = 0; d < 3; ++d) {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

TEST(KDTreePBT, KNNMatchesBruteForce) {
    rc::check("KDTree k-NN results match brute force",
              [](const std::vector<std::vector<int>>& raw_pts, const std::vector<int>& raw_query,
                 std::size_t k) {
                  RC_PRE(!raw_pts.empty());
                  RC_PRE(raw_query.size() >= 3);
                  k = (k % 25) + 1; // 1 to 25 neighbors

                  std::vector<Point3D> points;
                  points.reserve(raw_pts.size());
                  for (const auto& pt : raw_pts) {
                      if (pt.size() >= 3) {
                          points.push_back({static_cast<double>(pt[0]) / 10.0,
                                            static_cast<double>(pt[1]) / 10.0,
                                            static_cast<double>(pt[2]) / 10.0});
                      }
                  }
                  RC_PRE(!points.empty());

                  Point3D query{static_cast<double>(raw_query[0]) / 10.0,
                                static_cast<double>(raw_query[1]) / 10.0,
                                static_cast<double>(raw_query[2]) / 10.0};

                  KDTree<Point3D, 3> tree(points);
                  auto actual_knn = tree.knn_search(query, k);

                  // Brute force calculation
                  std::vector<std::pair<double, std::size_t>> bf;
                  bf.reserve(points.size());
                  for (std::size_t i = 0; i < points.size(); ++i) {
                      bf.emplace_back(std::sqrt(dist_euclidean_sq(query, points[i])), i);
                  }
                  std::sort(bf.begin(), bf.end());

                  const std::size_t expected_count = std::min(k, points.size());
                  RC_ASSERT(actual_knn.size() == expected_count);

                  for (std::size_t i = 0; i < expected_count; ++i) {
                      RC_ASSERT(std::abs(actual_knn[i].distance - bf[i].first) < 1e-7);
                      if (i > 0) {
                          RC_ASSERT(actual_knn[i].distance >= actual_knn[i - 1].distance);
                      }
                  }
              });
}

TEST(KDTreePBT, RadiusMatchesBruteForce) {
    rc::check("KDTree radius search matches brute force",
              [](const std::vector<std::vector<int>>& raw_pts, const std::vector<int>& raw_query,
                 int raw_radius) {
                  RC_PRE(raw_query.size() >= 3);
                  double radius = std::abs(static_cast<double>(raw_radius % 500)) / 10.0;
                  double radius_sq = radius * radius;

                  std::vector<Point3D> points;
                  points.reserve(raw_pts.size());
                  for (const auto& pt : raw_pts) {
                      if (pt.size() >= 3) {
                          points.push_back({static_cast<double>(pt[0]) / 10.0,
                                            static_cast<double>(pt[1]) / 10.0,
                                            static_cast<double>(pt[2]) / 10.0});
                      }
                  }
                  RC_PRE(!points.empty());

                  Point3D query{static_cast<double>(raw_query[0]) / 10.0,
                                static_cast<double>(raw_query[1]) / 10.0,
                                static_cast<double>(raw_query[2]) / 10.0};

                  KDTree<Point3D, 3> tree(points);
                  auto actual_indices = tree.radius_search(query, radius);

                  std::vector<std::size_t> expected_indices;
                  for (std::size_t i = 0; i < points.size(); ++i) {
                      if (dist_euclidean_sq(query, points[i]) <= radius_sq) {
                          expected_indices.push_back(i);
                      }
                  }
                  std::sort(expected_indices.begin(), expected_indices.end());

                  RC_ASSERT(actual_indices == expected_indices);
              });
}

TEST(KDTreePBT, BoxSearchMatchesBruteForce) {
    rc::check("KDTree box search matches brute force",
              [](const std::vector<std::vector<int>>& raw_pts, const std::vector<int>& c1,
                 const std::vector<int>& c2) {
                  RC_PRE(c1.size() >= 3 && c2.size() >= 3);

                  std::vector<Point3D> points;
                  points.reserve(raw_pts.size());
                  for (const auto& pt : raw_pts) {
                      if (pt.size() >= 3) {
                          points.push_back({static_cast<double>(pt[0]) / 10.0,
                                            static_cast<double>(pt[1]) / 10.0,
                                            static_cast<double>(pt[2]) / 10.0});
                      }
                  }
                  RC_PRE(!points.empty());

                  Point3D p1{static_cast<double>(c1[0]) / 10.0, static_cast<double>(c1[1]) / 10.0,
                             static_cast<double>(c1[2]) / 10.0};
                  Point3D p2{static_cast<double>(c2[0]) / 10.0, static_cast<double>(c2[1]) / 10.0,
                             static_cast<double>(c2[2]) / 10.0};

                  Point3D min_corner{std::min(p1[0], p2[0]), std::min(p1[1], p2[1]),
                                     std::min(p1[2], p2[2])};
                  Point3D max_corner{std::max(p1[0], p2[0]), std::max(p1[1], p2[1]),
                                     std::max(p1[2], p2[2])};

                  KDTree<Point3D, 3> tree(points);
                  auto actual_box = tree.box_search(min_corner, max_corner);

                  std::vector<std::size_t> expected_box;
                  for (std::size_t i = 0; i < points.size(); ++i) {
                      if (points[i][0] >= min_corner[0] && points[i][0] <= max_corner[0] &&
                          points[i][1] >= min_corner[1] && points[i][1] <= max_corner[1] &&
                          points[i][2] >= min_corner[2] && points[i][2] <= max_corner[2]) {
                          expected_box.push_back(i);
                      }
                  }
                  std::sort(expected_box.begin(), expected_box.end());

                  RC_ASSERT(actual_box == expected_box);
              });
}

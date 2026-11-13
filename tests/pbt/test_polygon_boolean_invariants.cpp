/**
 * @file test_polygon_boolean_invariants.cpp
 * @brief RapidCheck property-based tests and 10,000-shape GIS oracle verification for
 *        2D Polygon Boolean Operations.
 */

#include "algoat/geometry/polygon_boolean.hpp"
#include "algoat/geometry/predicates.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numbers>
#include <random>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

using namespace algoat::geometry;

namespace {

double total_area(const MultiPolygon<Point2D>& mp) {
    double sum = 0.0;
    for (const auto& p : mp) {
        sum += p.area();
    }
    return sum;
}

bool point_in_boolean_poly(const BooleanPolygon<Point2D>& poly, const Point2D& pt) {
    if (poly.outer_ring.size() < 3) {
        return false;
    }
    auto in_ring = [&](const std::vector<Point2D>& ring) {
        bool inside = false;
        const std::size_t n = ring.size();
        for (std::size_t i = 0; i < n; ++i) {
            const Point2D& a = ring[i];
            const Point2D& b = ring[(i + 1 == n) ? 0 : (i + 1)];
            if ((a.y > pt.y) != (b.y > pt.y)) {
                const double x_int = a.x + (pt.y - a.y) * (b.x - a.x) / (b.y - a.y);
                if (pt.x < x_int) {
                    inside = !inside;
                }
            }
        }
        return inside;
    };
    if (!in_ring(poly.outer_ring)) {
        return false;
    }
    for (const auto& h : poly.holes) {
        if (in_ring(h)) {
            return false;
        }
    }
    return true;
}

bool point_in_multipoly(const MultiPolygon<Point2D>& mp, const Point2D& pt) {
    for (const auto& p : mp) {
        if (point_in_boolean_poly(p, pt)) {
            return true;
        }
    }
    return false;
}

bool point_near_segments(const std::vector<Point2D>& ring, const Point2D& pt, double eps) {
    const std::size_t n = ring.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point2D& a = ring[i];
        const Point2D& b = ring[(i + 1 == n) ? 0 : (i + 1)];
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        const double len2 = dx * dx + dy * dy;
        double t = 0.0;
        if (len2 > 0.0) {
            t = std::clamp(((pt.x - a.x) * dx + (pt.y - a.y) * dy) / len2, 0.0, 1.0);
        }
        const double px = a.x + t * dx;
        const double py = a.y + t * dy;
        const double d2 = (pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py);
        if (d2 <= eps * eps) {
            return true;
        }
    }
    return false;
}

bool point_near_boundary(const BooleanPolygon<Point2D>& poly, const Point2D& pt, double eps) {
    if (point_near_segments(poly.outer_ring, pt, eps)) {
        return true;
    }
    for (const auto& h : poly.holes) {
        if (point_near_segments(h, pt, eps)) {
            return true;
        }
    }
    return false;
}

BooleanPolygon<Point2D> make_star_polygon(std::mt19937_64& rng, bool with_hole) {
    std::uniform_int_distribution<int> n_dist(5, 14);
    std::uniform_real_distribution<double> center_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> rad_dist(4.0, 14.0);
    std::uniform_int_distribution<int> grid_snap(0, 1);

    const int n = n_dist(rng);
    const double cx = grid_snap(rng) ? std::round(center_dist(rng)) : center_dist(rng);
    const double cy = grid_snap(rng) ? std::round(center_dist(rng)) : center_dist(rng);

    BooleanPolygon<Point2D> poly;
    for (int i = 0; i < n; ++i) {
        const double ang =
            (2.0 * std::numbers::pi * static_cast<double>(i)) / static_cast<double>(n);
        const double r = rad_dist(rng);
        double x = cx + r * std::cos(ang);
        double y = cy + r * std::sin(ang);
        if (grid_snap(rng)) {
            x = std::round(x * 4.0) * 0.25;
            y = std::round(y * 4.0) * 0.25;
        }
        poly.outer_ring.push_back({x, y});
    }

    if (with_hole) {
        std::uniform_real_distribution<double> hole_r(0.8, 2.2);
        const double hr = hole_r(rng);
        poly.holes.push_back({
            {cx - hr, cy - hr},
            {cx - hr, cy + hr},
            {cx + hr, cy + hr},
            {cx + hr, cy - hr},
        });
    }
    return poly;
}

rc::Gen<BooleanPolygon<Point2D>> gen_star_or_grid_poly() {
    return rc::gen::exec([]() {
        const bool grid_mode = *rc::gen::arbitrary<bool>();
        BooleanPolygon<Point2D> poly;
        if (grid_mode) {
            // Integer grid orthogonal star polygon to trigger heavy collinear/vertical overlaps
            const int n = *rc::gen::inRange(4, 12);
            const double cx = static_cast<double>(*rc::gen::inRange(-4, 5));
            const double cy = static_cast<double>(*rc::gen::inRange(-4, 5));
            for (int i = 0; i < n; ++i) {
                const double ang =
                    (2.0 * std::numbers::pi * static_cast<double>(i)) / static_cast<double>(n);
                const double r = static_cast<double>(*rc::gen::inRange(2, 9));
                poly.outer_ring.push_back(
                    {std::round(cx + r * std::cos(ang)), std::round(cy + r * std::sin(ang))});
            }
        } else {
            const int n = *rc::gen::inRange(4, 14);
            const double cx = static_cast<double>(*rc::gen::inRange(-20, 21)) * 0.5;
            const double cy = static_cast<double>(*rc::gen::inRange(-20, 21)) * 0.5;
            for (int i = 0; i < n; ++i) {
                const double ang =
                    (2.0 * std::numbers::pi * static_cast<double>(i)) / static_cast<double>(n);
                const double r = static_cast<double>(*rc::gen::inRange(3, 18)) * 0.5;
                poly.outer_ring.push_back({cx + r * std::cos(ang), cy + r * std::sin(ang)});
            }
            if (*rc::gen::inRange(0, 4) == 0) {
                poly.holes.push_back({
                    {cx - 0.5, cy - 0.5},
                    {cx - 0.5, cy + 0.5},
                    {cx + 0.5, cy + 0.5},
                    {cx + 0.5, cy - 0.5},
                });
            }
        }
        return poly;
    });
}

void verify_orientation_invariants(const MultiPolygon<Point2D>& mp) {
    for (const auto& p : mp) {
        RC_ASSERT(p.outer_ring.size() >= 3u);
        RC_ASSERT(detail::ring_signed_area<Point2D>(p.outer_ring) > 0.0);
        for (const auto& h : p.holes) {
            RC_ASSERT(h.size() >= 3u);
            RC_ASSERT(detail::ring_signed_area<Point2D>(h) < 0.0);
        }
    }
}

} // namespace

RC_GTEST_PROP(PolygonBooleanPBT, AlgebraicSetAreaIdentitiesAndOrientation, ()) {
    const auto a = *gen_star_or_grid_poly();
    const auto b = *gen_star_or_grid_poly();

    const auto mp_inter = polygon_intersection(a, b);
    const auto mp_union = polygon_union(a, b);
    const auto mp_diff_ab = polygon_difference(a, b);
    const auto mp_diff_ba = polygon_difference(b, a);
    const auto mp_xor = polygon_xor(a, b);

    verify_orientation_invariants(mp_inter);
    verify_orientation_invariants(mp_union);
    verify_orientation_invariants(mp_diff_ab);
    verify_orientation_invariants(mp_diff_ba);
    verify_orientation_invariants(mp_xor);

    const double area_a = a.area();
    const double area_b = b.area();
    const double ai = total_area(mp_inter);
    const double au = total_area(mp_union);
    const double ad_ab = total_area(mp_diff_ab);
    const double ad_ba = total_area(mp_diff_ba);
    const double ax = total_area(mp_xor);

    const double tol = 1e-6 * std::max({1.0, area_a, area_b});
    RC_ASSERT(std::abs((au + ai) - (area_a + area_b)) <= tol);
    RC_ASSERT(std::abs((ad_ab + ai) - area_a) <= tol);
    RC_ASSERT(std::abs((ad_ba + ai) - area_b) <= tol);
    RC_ASSERT(std::abs(ax - (au - ai)) <= tol);
    RC_ASSERT(std::abs(ax - (ad_ab + ad_ba)) <= tol);
}

RC_GTEST_PROP(PolygonBooleanPBT, CommutativitySelfLawsAndPointInPolygonOracle, ()) {
    const auto a = *gen_star_or_grid_poly();
    const auto b = *gen_star_or_grid_poly();

    const double tol = 1e-6 * std::max({1.0, a.area(), b.area()});

    // Commutativity of areas
    RC_ASSERT(std::abs(total_area(polygon_intersection(a, b)) -
                       total_area(polygon_intersection(b, a))) <= tol);
    RC_ASSERT(std::abs(total_area(polygon_union(a, b)) - total_area(polygon_union(b, a))) <= tol);
    RC_ASSERT(std::abs(total_area(polygon_xor(a, b)) - total_area(polygon_xor(b, a))) <= tol);

    // Self identities
    RC_ASSERT(std::abs(total_area(polygon_intersection(a, a)) - a.area()) <= tol);
    RC_ASSERT(std::abs(total_area(polygon_union(a, a)) - a.area()) <= tol);
    RC_ASSERT(polygon_difference(a, a).empty());
    RC_ASSERT(polygon_xor(a, a).empty());

    // Point-in-polygon differential oracle away from boundaries
    const auto mp_inter = polygon_intersection(a, b);
    const auto mp_union = polygon_union(a, b);
    const auto mp_diff = polygon_difference(a, b);
    const auto mp_xor = polygon_xor(a, b);

    for (int gx = -12; gx <= 12; gx += 3) {
        for (int gy = -12; gy <= 12; gy += 3) {
            const Point2D pt{static_cast<double>(gx) + 0.137, static_cast<double>(gy) + 0.293};
            if (point_near_boundary(a, pt, 1e-4) || point_near_boundary(b, pt, 1e-4)) {
                continue;
            }
            const bool in_a = point_in_boolean_poly(a, pt);
            const bool in_b = point_in_boolean_poly(b, pt);
            RC_ASSERT(point_in_multipoly(mp_inter, pt) == (in_a && in_b));
            RC_ASSERT(point_in_multipoly(mp_union, pt) == (in_a || in_b));
            RC_ASSERT(point_in_multipoly(mp_diff, pt) == (in_a && !in_b));
            RC_ASSERT(point_in_multipoly(mp_xor, pt) == (in_a != in_b));
        }
    }
}

TEST(PolygonBooleanOracle, Randomized10000GisShapesAlgebraicAndPipConsistency) {
    std::mt19937_64 rng(20261114ULL);
    constexpr int kTotalShapes = 10000;

    for (int iter = 0; iter < kTotalShapes; ++iter) {
        const bool hole_a = (iter % 4 == 0);
        const bool hole_b = (iter % 5 == 0);
        const auto a = make_star_polygon(rng, hole_a);
        const auto b = make_star_polygon(rng, hole_b);

        const auto mp_i = polygon_intersection(a, b);
        const auto mp_u = polygon_union(a, b);
        const auto mp_d = polygon_difference(a, b);
        const auto mp_x = polygon_xor(a, b);

        const double aa = a.area();
        const double ab = b.area();
        const double ai = total_area(mp_i);
        const double au = total_area(mp_u);
        const double ad = total_area(mp_d);
        const double ax = total_area(mp_x);
        const double tol = 1e-7 * std::max({1.0, aa, ab});

        ASSERT_NEAR(au + ai, aa + ab, tol) << "iter=" << iter;
        ASSERT_NEAR(ad + ai, aa, tol) << "iter=" << iter;
        ASSERT_NEAR(ax, au - ai, tol) << "iter=" << iter;

        for (const auto& poly : mp_u) {
            ASSERT_GE(poly.outer_ring.size(), 3u) << "iter=" << iter;
            ASSERT_GT(detail::ring_signed_area<Point2D>(poly.outer_ring), 0.0) << "iter=" << iter;
            for (const auto& h : poly.holes) {
                ASSERT_GE(h.size(), 3u) << "iter=" << iter;
                ASSERT_LT(detail::ring_signed_area<Point2D>(h), 0.0) << "iter=" << iter;
            }
        }

        // Sample deterministic probe point for PIP oracle check on every shape pair
        const Point2D probe{1.317, -2.419};
        if (!point_near_boundary(a, probe, 1e-5) && !point_near_boundary(b, probe, 1e-5)) {
            const bool ia = point_in_boolean_poly(a, probe);
            const bool ib = point_in_boolean_poly(b, probe);
            ASSERT_EQ(point_in_multipoly(mp_i, probe), ia && ib) << "iter=" << iter;
            ASSERT_EQ(point_in_multipoly(mp_u, probe), ia || ib) << "iter=" << iter;
            ASSERT_EQ(point_in_multipoly(mp_d, probe), ia && !ib) << "iter=" << iter;
            ASSERT_EQ(point_in_multipoly(mp_x, probe), ia != ib) << "iter=" << iter;
        }
    }
}

#include "algoat/geometry/kdtree.hpp"
#include "algoat/geometry/predicates.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace algoat::geometry;

static_assert(SpatialPoint<Point2D, 2>);
static_assert(SpatialPoint<Point3D, 3>);

TEST(PredicatesTest, TwoSumAndTwoProductFMA) {
    double s = 0.0;
    double e = 0.0;
    two_sum(1.0, 1e-16, s, e);
    EXPECT_DOUBLE_EQ(s + e, 1.0 + 1e-16);

    double p = 0.0;
    two_product_fma(1.0 + 1e-16, 1.0 - 1e-16, p, e);
    EXPECT_DOUBLE_EQ(p + e, (1.0 + 1e-16) * (1.0 - 1e-16));
}

// ============================================================================
// 1. orient2d: Basic Tests & Constexpr Verification
// ============================================================================

TEST(PredicatesTest, Orient2DBasic) {
    Point2D a{0.0, 0.0};
    Point2D b{1.0, 0.0};
    Point2D c{0.0, 1.0}; // (0,0) -> (1,0) -> (0,1) is CounterClockwise

    EXPECT_EQ(orient2d(a, b, c), Orientation::CounterClockwise);
    EXPECT_EQ(orient2d(a, c, b), Orientation::Clockwise);

    Point2D d{2.0, 0.0}; // (0,0), (1,0), (2,0) are Collinear
    EXPECT_EQ(orient2d(a, b, d), Orientation::Collinear);
}

TEST(PredicatesTest, Orient2DCoincident) {
    Point2D a{1.5, 2.5};
    Point2D b{3.0, 4.0};

    // Any coincident points must yield Collinear
    EXPECT_EQ(orient2d(a, a, b), Orientation::Collinear);
    EXPECT_EQ(orient2d(a, b, a), Orientation::Collinear);
    EXPECT_EQ(orient2d(b, a, a), Orientation::Collinear);
    EXPECT_EQ(orient2d(a, a, a), Orientation::Collinear);
}

TEST(PredicatesTest, Orient2DConstexpr) {
    constexpr Point2D a{0.0, 0.0};
    constexpr Point2D b{1.0, 0.0};
    constexpr Point2D c{0.0, 1.0};
    constexpr Point2D d{2.0, 0.0};

    constexpr auto ccw = orient2d(a, b, c);
    constexpr auto cw = orient2d(a, c, b);
    constexpr auto col = orient2d(a, b, d);

    static_assert(ccw == Orientation::CounterClockwise);
    static_assert(cw == Orientation::Clockwise);
    static_assert(col == Orientation::Collinear);

    EXPECT_EQ(ccw, Orientation::CounterClockwise);
    EXPECT_EQ(cw, Orientation::Clockwise);
    EXPECT_EQ(col, Orientation::Collinear);
}

// ============================================================================
// 2. Pathological Near-Collinear Datasets (Kettner et al. & Shewchuk)
// ============================================================================

TEST(PredicatesTest, Orient2DKettnerClassroomGrid) {
    // Kettner, Mehlhorn, Pion, Schirra, Yap (2008) "Classroom Examples of Robustness Problems"
    // Line defined by q=(12, 12) and r=(24, 24).
    // Test points near p1=(0.5, 0.5): p(X, Y) = (0.5 + X * 2^-53, 0.5 + Y * 2^-53).
    // Mathematically:
    // If X == Y, point is exactly on the line (Collinear).
    // If Y > X, point is CounterClockwise (left of line qr).
    // If Y < X, point is Clockwise (right of line qr).
    const Point2D q{12.0, 12.0};
    const Point2D r{24.0, 24.0};

    constexpr double u = 1.11022302462515654042e-16; // 2^-53

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            Point2D p{0.5 + x * u, 0.5 + y * u};
            Orientation expected;
            if (x == y) {
                expected = Orientation::Collinear;
            } else if (y > x) {
                expected = Orientation::CounterClockwise;
            } else {
                expected = Orientation::Clockwise;
            }

            Orientation result = orient2d(q, r, p);
            EXPECT_EQ(result, expected) << "Mismatch at x=" << x << ", y=" << y;
        }
    }
}

// ============================================================================
// 3. orient3d: 3D Orientation Tests
// ============================================================================

TEST(PredicatesTest, Orient3DBasic) {
    Point3D a{0.0, 0.0, 0.0};
    Point3D b{1.0, 0.0, 0.0};
    Point3D c{0.0, 1.0, 0.0};
    Point3D d_above{0.0, 0.0, 1.0};
    Point3D d_below{0.0, 0.0, -1.0};
    Point3D d_coplanar{0.5, 0.5, 0.0};

    // Right-hand rule / Shewchuk determinant sign
    EXPECT_EQ(orient3d(a, b, c, d_above), Orientation::Clockwise);
    EXPECT_EQ(orient3d(a, b, c, d_below), Orientation::CounterClockwise);
    EXPECT_EQ(orient3d(a, b, c, d_coplanar), Orientation::Collinear);
}

TEST(PredicatesTest, Orient3DCoincident) {
    Point3D a{1.0, 2.0, 3.0};
    Point3D b{4.0, 5.0, 6.0};
    Point3D c{7.0, 8.0, 9.0};

    EXPECT_EQ(orient3d(a, a, b, c), Orientation::Collinear);
    EXPECT_EQ(orient3d(a, b, a, c), Orientation::Collinear);
    EXPECT_EQ(orient3d(a, b, c, a), Orientation::Collinear);
    EXPECT_EQ(orient3d(a, b, c, b), Orientation::Collinear);
}

TEST(PredicatesTest, Orient3DConstexpr) {
    constexpr Point3D a{0.0, 0.0, 0.0};
    constexpr Point3D b{1.0, 0.0, 0.0};
    constexpr Point3D c{0.0, 1.0, 0.0};
    constexpr Point3D d_above{0.0, 0.0, 1.0};
    constexpr Point3D d_coplanar{0.5, 0.5, 0.0};

    constexpr auto cw = orient3d(a, b, c, d_above);
    constexpr auto col = orient3d(a, b, c, d_coplanar);

    static_assert(cw == Orientation::Clockwise);
    static_assert(col == Orientation::Collinear);

    EXPECT_EQ(cw, Orientation::Clockwise);
    EXPECT_EQ(col, Orientation::Collinear);
}

// ============================================================================
// 4. incircle: 2D Circumcircle Predicate Tests
// ============================================================================

TEST(PredicatesTest, IncircleBasic) {
    // Triangle (pa, pb, pc) oriented CounterClockwise: (0,0), (1,0), (0,1)
    Point2D a{0.0, 0.0};
    Point2D b{1.0, 0.0};
    Point2D c{0.0, 1.0};

    Point2D d_inside{0.2, 0.2};
    Point2D d_cocircular{1.0, 1.0}; // Hypotenuse diagonal, circumcircle passes through (1,1)
    Point2D d_outside{2.0, 2.0};

    EXPECT_EQ(incircle(a, b, c, d_inside), CircleLocation::Inside);
    EXPECT_EQ(incircle(a, b, c, d_cocircular), CircleLocation::Cocirular);
    EXPECT_EQ(incircle(a, b, c, d_outside), CircleLocation::Outside);
}

TEST(PredicatesTest, IncircleCoincident) {
    Point2D a{0.0, 0.0};
    Point2D b{1.0, 0.0};
    Point2D c{0.0, 1.0};

    // Any query coincident with a triangle vertex lies on the circle
    EXPECT_EQ(incircle(a, b, c, a), CircleLocation::Cocirular);
    EXPECT_EQ(incircle(a, b, c, b), CircleLocation::Cocirular);
    EXPECT_EQ(incircle(a, b, c, c), CircleLocation::Cocirular);
}

TEST(PredicatesTest, IncircleConstexpr) {
    constexpr Point2D a{0.0, 0.0};
    constexpr Point2D b{1.0, 0.0};
    constexpr Point2D c{0.0, 1.0};
    constexpr Point2D d_in{0.2, 0.2};
    constexpr Point2D d_cocirc{1.0, 1.0};
    constexpr Point2D d_out{2.0, 2.0};

    constexpr auto inside_loc = incircle(a, b, c, d_in);
    constexpr auto cocirc_loc = incircle(a, b, c, d_cocirc);
    constexpr auto outside_loc = incircle(a, b, c, d_out);

    static_assert(inside_loc == CircleLocation::Inside);
    static_assert(cocirc_loc == CircleLocation::Cocirular);
    static_assert(outside_loc == CircleLocation::Outside);

    EXPECT_EQ(inside_loc, CircleLocation::Inside);
    EXPECT_EQ(cocirc_loc, CircleLocation::Cocirular);
    EXPECT_EQ(outside_loc, CircleLocation::Outside);
}

TEST(PredicatesTest, IncircleEpsilonPerturbation) {
    // Circle centered at origin with radius R = 1.0.
    // Points at angles 0, 2pi/3, 4pi/3 (equilateral triangle)
    const double r = 1.0;
    Point2D a{r * 1.0, r * 0.0};
    Point2D b{r * -0.5, r * (std::sqrt(3.0) / 2.0)};
    Point2D c{r * -0.5, r * (-std::sqrt(3.0) / 2.0)};

    constexpr double eps = 1.11022302462515654042e-16;

    // Test a point at angle pi/3 perturbed slightly inside or outside
    Point2D p_inside{(r - 10.0 * eps) * 0.5, (r - 10.0 * eps) * (std::sqrt(3.0) / 2.0)};
    Point2D p_outside{(r + 10.0 * eps) * 0.5, (r + 10.0 * eps) * (std::sqrt(3.0) / 2.0)};

    EXPECT_EQ(incircle(a, b, c, p_inside), CircleLocation::Inside);
    EXPECT_EQ(incircle(a, b, c, p_outside), CircleLocation::Outside);
}

// ============================================================================
// 5. insphere: 3D Circumsphere Predicate Tests
// ============================================================================

TEST(PredicatesTest, InsphereBasic) {
    Point3D a{0.0, 0.0, 0.0};
    Point3D b{1.0, 0.0, 0.0};
    Point3D c{0.0, 1.0, 0.0};
    Point3D d{-0.0, 0.0, 1.0}; // Orient3D(a,b,c,d) is Clockwise (-1)

    // Note: insphere orientation depends on orient3d sign.
    // For standard positive orientation, inside gives Inside.
    Point3D e_inside{0.2, 0.2, 0.2};
    Point3D e_cospherical{1.0, 1.0, 1.0};
    Point3D e_outside{3.0, 3.0, 3.0};

    // Verify determinism and non-coincident evaluation
    CircleLocation loc_in = insphere(a, b, c, d, e_inside);
    CircleLocation loc_sph = insphere(a, b, c, d, e_cospherical);
    CircleLocation loc_out = insphere(a, b, c, d, e_outside);

    EXPECT_EQ(loc_sph, CircleLocation::Cocirular);
    EXPECT_NE(loc_in, loc_out);
}

TEST(PredicatesTest, InsphereCoincident) {
    Point3D a{0.0, 0.0, 0.0};
    Point3D b{1.0, 0.0, 0.0};
    Point3D c{0.0, 1.0, 0.0};
    Point3D d{0.0, 0.0, 1.0};

    EXPECT_EQ(insphere(a, b, c, d, a), CircleLocation::Cocirular);
    EXPECT_EQ(insphere(a, b, c, d, b), CircleLocation::Cocirular);
    EXPECT_EQ(insphere(a, b, c, d, c), CircleLocation::Cocirular);
    EXPECT_EQ(insphere(a, b, c, d, d), CircleLocation::Cocirular);
}

TEST(PredicatesTest, InsphereConstexpr) {
    constexpr Point3D a{0.0, 0.0, 0.0};
    constexpr Point3D b{1.0, 0.0, 0.0};
    constexpr Point3D c{0.0, 1.0, 0.0};
    constexpr Point3D d{0.0, 0.0, 1.0};
    constexpr Point3D e_cospherical{1.0, 1.0, 1.0};

    constexpr auto loc = insphere(a, b, c, d, e_cospherical);
    static_assert(loc == CircleLocation::Cocirular);
    EXPECT_EQ(loc, CircleLocation::Cocirular);
}

// ============================================================================
// 6. Large Pathological Dataset Stress Test (10^6 Near-Collinear Points)
// ============================================================================

TEST(PredicatesTest, PathologicalNearCollinearMillionPoints) {
    // 10^6 evaluations along y = x line with perturbations down to machine epsilon
    const Point2D q{12.0, 12.0};
    const Point2D r{24.0, 24.0};

    constexpr double eps = 1.11022302462515654042e-16; // 2^-53

    constexpr std::size_t num_points = 1'000'000;
    std::size_t collinear_count = 0;
    std::size_t ccw_count = 0;
    std::size_t cw_count = 0;

    for (std::size_t i = 0; i < num_points; ++i) {
        // Vary the base coordinate across exponents and dyadic scales
        double base = 0.5 + static_cast<double>(i % 1000) * 1e-4;

        // Cyclic perturbation pattern: 0, +eps, -eps, +2*eps, -2*eps, +10*eps, -10*eps
        int mode = static_cast<int>(i % 7);
        double delta = 0.0;
        Orientation expected = Orientation::Collinear;

        switch (mode) {
        case 0:
            delta = 0.0;
            expected = Orientation::Collinear;
            break;
        case 1:
            delta = eps;
            expected = Orientation::CounterClockwise;
            break;
        case 2:
            delta = -eps;
            expected = Orientation::Clockwise;
            break;
        case 3:
            delta = 2.0 * eps;
            expected = Orientation::CounterClockwise;
            break;
        case 4:
            delta = -2.0 * eps;
            expected = Orientation::Clockwise;
            break;
        case 5:
            delta = 10.0 * eps;
            expected = Orientation::CounterClockwise;
            break;
        case 6:
            delta = -10.0 * eps;
            expected = Orientation::Clockwise;
            break;
        }

        Point2D p{base, base + delta};
        Orientation result = orient2d(q, r, p);
        ASSERT_EQ(result, expected) << "Failed at index " << i << " with delta=" << delta;

        if (result == Orientation::Collinear)
            ++collinear_count;
        else if (result == Orientation::CounterClockwise)
            ++ccw_count;
        else
            ++cw_count;
    }

    EXPECT_GT(collinear_count, 0u);
    EXPECT_GT(ccw_count, 0u);
    EXPECT_GT(cw_count, 0u);
}

TEST(PredicatesTest, PathologicalNearCocircularMillionPoints) {
    // Exact circle centered at origin (0, 0) with radius R = 25.0
    // Triangle (a, b, c) in counter-clockwise order:
    const Point2D a{25.0, 0.0};
    const Point2D b{0.0, 25.0};
    const Point2D c{-25.0, 0.0};

    // Exact Pythagorean integer points on x^2 + y^2 = 25^2 = 625 with y < 0:
    const std::vector<Point2D> exact_base_points = {{0.0, -25.0},   {7.0, -24.0},   {15.0, -20.0},
                                                    {20.0, -15.0},  {24.0, -7.0},   {-7.0, -24.0},
                                                    {-15.0, -20.0}, {-20.0, -15.0}, {-24.0, -7.0}};

    constexpr double eps = detail::epsilon;
    constexpr std::size_t num_points = 1'000'000;

    std::size_t cocircular_count = 0;
    std::size_t inside_count = 0;
    std::size_t outside_count = 0;

    for (std::size_t i = 0; i < num_points; ++i) {
        const auto& base = exact_base_points[i % exact_base_points.size()];

        int mode = static_cast<int>(i % 7);
        double y_perturbed = base.y;
        CircleLocation expected = CircleLocation::Cocirular;

        switch (mode) {
        case 0:
            y_perturbed = base.y;
            expected = CircleLocation::Cocirular;
            break;
        case 1:
            y_perturbed = std::nextafter(base.y, 0.0);
            expected = CircleLocation::Inside;
            break;
        case 2:
            y_perturbed = std::nextafter(base.y, -100.0);
            expected = CircleLocation::Outside;
            break;
        case 3: {
            double y1 = std::nextafter(base.y, 0.0);
            y_perturbed = std::nextafter(y1, 0.0);
            expected = CircleLocation::Inside;
            break;
        }
        case 4: {
            double y1 = std::nextafter(base.y, -100.0);
            y_perturbed = std::nextafter(y1, -100.0);
            expected = CircleLocation::Outside;
            break;
        }
        case 5: {
            double yk = base.y;
            for (int k = 0; k < 5; ++k) {
                yk = std::nextafter(yk, 0.0);
            }
            y_perturbed = yk;
            expected = CircleLocation::Inside;
            break;
        }
        case 6: {
            double yk = base.y;
            for (int k = 0; k < 5; ++k) {
                yk = std::nextafter(yk, -100.0);
            }
            y_perturbed = yk;
            expected = CircleLocation::Outside;
            break;
        }
        }

        // Because base.y < 0, perturbing towards 0 moves point strictly inside the circle,
        // and perturbing towards -100 moves point strictly outside the circle.
        Point2D p{base.x, y_perturbed};
        CircleLocation result = incircle(a, b, c, p);
        ASSERT_EQ(result, expected) << "Failed at index " << i << " with base=(" << base.x << ","
                                    << base.y << ") y_perturbed=" << y_perturbed;

        if (result == CircleLocation::Cocirular) {
            ++cocircular_count;
        } else if (result == CircleLocation::Inside) {
            ++inside_count;
        } else {
            ++outside_count;
        }
    }

    EXPECT_GT(cocircular_count, 0u);
    EXPECT_GT(inside_count, 0u);
    EXPECT_GT(outside_count, 0u);
}

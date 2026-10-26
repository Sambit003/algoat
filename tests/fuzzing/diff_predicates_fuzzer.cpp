#include "algoat/geometry/predicates.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

using namespace algoat::geometry;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // 5 Point2D (10 doubles) + 5 Point3D (15 doubles) = 25 doubles = 200 bytes
    constexpr size_t min_bytes = 25 * sizeof(double);
    if (size < min_bytes) {
        return 0;
    }

    double buf[25];
    std::memcpy(buf, data, sizeof(buf));

    for (int i = 0; i < 25; ++i) {
        if (!std::isfinite(buf[i]) || std::abs(buf[i]) > 1e30) {
            return 0;
        }
        if (std::abs(buf[i]) > 0.0 && std::abs(buf[i]) < 1e-150) {
            return 0;
        }
    }

    Point2D p2[5] = {
        {buf[0], buf[1]}, {buf[2], buf[3]}, {buf[4], buf[5]}, {buf[6], buf[7]}, {buf[8], buf[9]}};

    Point3D p3[5] = {{buf[10], buf[11], buf[12]},
                     {buf[13], buf[14], buf[15]},
                     {buf[16], buf[17], buf[18]},
                     {buf[19], buf[20], buf[21]},
                     {buf[22], buf[23], buf[24]}};

    // =========================================================================
    // 1. Orient2D: Permutation Parity & Coincidence
    // =========================================================================
    Orientation o2_abc = orient2d(p2[0], p2[1], p2[2]);
    Orientation o2_bac = orient2d(p2[1], p2[0], p2[2]);
    Orientation o2_bca = orient2d(p2[1], p2[2], p2[0]);
    Orientation o2_cab = orient2d(p2[2], p2[0], p2[1]);

    // Single transposition negates sign
    assert(static_cast<int>(o2_abc) == -static_cast<int>(o2_bac));
    // Even cyclic shifts preserve sign
    assert(static_cast<int>(o2_abc) == static_cast<int>(o2_bca));
    assert(static_cast<int>(o2_abc) == static_cast<int>(o2_cab));

    // Coincidence degeneracy
    assert(orient2d(p2[0], p2[0], p2[1]) == Orientation::Collinear);
    assert(orient2d(p2[0], p2[1], p2[0]) == Orientation::Collinear);
    assert(orient2d(p2[1], p2[0], p2[0]) == Orientation::Collinear);

    // =========================================================================
    // 2. Transitivity Assertions (Knuth CCW Axioms 4 & 5 + Collinear Ray Transitivity)
    // =========================================================================
    // Collinear Ray Transitivity: Points P1=(1,1), P2=(2,2), P3=(4,4) are strictly
    // ordered along the diagonal y = x. For ANY arbitrary query point Q, the
    // orientation relative to directed segments P1->P2, P2->P3, and P1->P3 must
    // be strictly transitive and equal: (P1 < P2 < P3) ==> orient(P1,P2,Q) == orient(P2,P3,Q) ==
    // orient(P1,P3,Q).
    const Point2D ray_p1{1.0, 1.0};
    const Point2D ray_p2{2.0, 2.0};
    const Point2D ray_p3{4.0, 4.0};
    Orientation o_p12 = orient2d(ray_p1, ray_p2, p2[0]);
    Orientation o_p23 = orient2d(ray_p2, ray_p3, p2[0]);
    Orientation o_p13 = orient2d(ray_p1, ray_p3, p2[0]);
    assert(o_p12 == o_p23);
    assert(o_p12 == o_p13);

    // Knuth Axiom 4 (Interiority):
    // (tbc > 0 && atc > 0 && abt > 0) ==> abc > 0
    // (tbc < 0 && atc < 0 && abt < 0) ==> abc < 0
    Orientation tbc = orient2d(p2[3], p2[1], p2[2]);
    Orientation atc = orient2d(p2[0], p2[3], p2[2]);
    Orientation abt = orient2d(p2[0], p2[1], p2[3]);
    if (tbc == Orientation::CounterClockwise && atc == Orientation::CounterClockwise &&
        abt == Orientation::CounterClockwise) {
        assert(orient2d(p2[0], p2[1], p2[2]) == Orientation::CounterClockwise);
    }
    if (tbc == Orientation::Clockwise && atc == Orientation::Clockwise &&
        abt == Orientation::Clockwise) {
        assert(orient2d(p2[0], p2[1], p2[2]) == Orientation::Clockwise);
    }

    // Knuth Axiom 5 (Transitivity among 5 points sharing pivot vertex t = p2[4]):
    // (tsp && tsq && tsr && tpq && tqr) ==> tpr
    const Point2D& t = p2[4];
    const Point2D& s = p2[3];
    const Point2D& p = p2[0];
    const Point2D& q = p2[1];
    const Point2D& r = p2[2];

    bool tsp = (orient2d(t, s, p) == Orientation::CounterClockwise);
    bool tsq = (orient2d(t, s, q) == Orientation::CounterClockwise);
    bool tsr = (orient2d(t, s, r) == Orientation::CounterClockwise);
    bool tpq = (orient2d(t, p, q) == Orientation::CounterClockwise);
    bool tqr = (orient2d(t, q, r) == Orientation::CounterClockwise);

    if (tsp && tsq && tsr && tpq && tqr) {
        assert(orient2d(t, p, r) == Orientation::CounterClockwise);
    }

    // =========================================================================
    // 3. Orient3D: Permutation Parity & Coincidence
    // =========================================================================
    Orientation o3_abcd = orient3d(p3[0], p3[1], p3[2], p3[3]);
    Orientation o3_bacd = orient3d(p3[1], p3[0], p3[2], p3[3]);
    Orientation o3_acbd = orient3d(p3[0], p3[2], p3[1], p3[3]);
    Orientation o3_abdc = orient3d(p3[0], p3[1], p3[3], p3[2]);

    assert(static_cast<int>(o3_abcd) == -static_cast<int>(o3_bacd));
    assert(static_cast<int>(o3_abcd) == -static_cast<int>(o3_acbd));
    assert(static_cast<int>(o3_abcd) == -static_cast<int>(o3_abdc));

    assert(orient3d(p3[0], p3[0], p3[1], p3[2]) == Orientation::Collinear);
    assert(orient3d(p3[0], p3[1], p3[0], p3[2]) == Orientation::Collinear);
    assert(orient3d(p3[0], p3[1], p3[2], p3[0]) == Orientation::Collinear);

    // =========================================================================
    // 4. InCircle: Permutation Parity & Coincidence
    // =========================================================================
    CircleLocation inc_abcd = incircle(p2[0], p2[1], p2[2], p2[3]);
    CircleLocation inc_bacd = incircle(p2[1], p2[0], p2[2], p2[3]);
    CircleLocation inc_bcda = incircle(p2[1], p2[2], p2[3], p2[0]);
    CircleLocation inc_cdab = incircle(p2[2], p2[3], p2[0], p2[1]);

    // Odd permutations negate sign
    assert(static_cast<int>(inc_abcd) == -static_cast<int>(inc_bacd));
    assert(static_cast<int>(inc_abcd) == -static_cast<int>(inc_bcda));
    // Even permutation preserves sign
    assert(static_cast<int>(inc_abcd) == static_cast<int>(inc_cdab));

    assert(incircle(p2[0], p2[1], p2[2], p2[0]) == CircleLocation::Cocirular);
    assert(incircle(p2[0], p2[1], p2[2], p2[1]) == CircleLocation::Cocirular);
    assert(incircle(p2[0], p2[1], p2[2], p2[2]) == CircleLocation::Cocirular);

    // =========================================================================
    // 5. InSphere: Permutation Parity & Coincidence
    // =========================================================================
    CircleLocation isp_abcde = insphere(p3[0], p3[1], p3[2], p3[3], p3[4]);
    CircleLocation isp_bacde = insphere(p3[1], p3[0], p3[2], p3[3], p3[4]);
    CircleLocation isp_bcdea = insphere(p3[1], p3[2], p3[3], p3[4], p3[0]);

    assert(static_cast<int>(isp_abcde) == -static_cast<int>(isp_bacde));
    assert(static_cast<int>(isp_abcde) == static_cast<int>(isp_bcdea));

    assert(insphere(p3[0], p3[1], p3[2], p3[3], p3[0]) == CircleLocation::Cocirular);
    assert(insphere(p3[0], p3[1], p3[2], p3[3], p3[1]) == CircleLocation::Cocirular);
    assert(insphere(p3[0], p3[1], p3[2], p3[3], p3[2]) == CircleLocation::Cocirular);
    assert(insphere(p3[0], p3[1], p3[2], p3[3], p3[3]) == CircleLocation::Cocirular);

    return 0;
}

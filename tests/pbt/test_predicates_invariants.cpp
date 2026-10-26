#include "algoat/geometry/predicates.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <rapidcheck.h>

using namespace algoat::geometry;

inline int sign_to_int(Orientation o) {
    return static_cast<int>(o);
}

inline int sign_to_int(CircleLocation c) {
    return static_cast<int>(c);
}

namespace rc {

template <> struct Arbitrary<Point2D> {
    static Gen<Point2D> arbitrary() {
        return gen::map(
            gen::tuple(gen::inRange<int>(-1000000, 1000000), gen::inRange<int>(-1000000, 1000000)),
            [](const std::tuple<int, int>& t) {
                return Point2D{static_cast<double>(std::get<0>(t)) / 10.0,
                               static_cast<double>(std::get<1>(t)) / 10.0};
            });
    }
};

template <> struct Arbitrary<Point3D> {
    static Gen<Point3D> arbitrary() {
        return gen::map(gen::tuple(gen::inRange<int>(-1000000, 1000000),
                                   gen::inRange<int>(-1000000, 1000000),
                                   gen::inRange<int>(-1000000, 1000000)),
                        [](const std::tuple<int, int, int>& t) {
                            return Point3D{static_cast<double>(std::get<0>(t)) / 10.0,
                                           static_cast<double>(std::get<1>(t)) / 10.0,
                                           static_cast<double>(std::get<2>(t)) / 10.0};
                        });
    }
};

} // namespace rc

TEST(PredicatesPBT, Orient2DAlternatingAndCyclic) {
    rc::check("orient2d satisfies antisymmetry and cyclic invariance",
              [](const Point2D& a, const Point2D& b, const Point2D& c) {
                  RC_PRE(std::isfinite(a.x) && std::isfinite(a.y));
                  RC_PRE(std::isfinite(b.x) && std::isfinite(b.y));
                  RC_PRE(std::isfinite(c.x) && std::isfinite(c.y));

                  int s_abc = sign_to_int(orient2d(a, b, c));
                  int s_bac = sign_to_int(orient2d(b, a, c));
                  int s_bca = sign_to_int(orient2d(b, c, a));
                  int s_cab = sign_to_int(orient2d(c, a, b));

                  // Swapping two points negates sign
                  RC_ASSERT(s_abc == -s_bac);
                  // Cyclic permutations preserve sign
                  RC_ASSERT(s_abc == s_bca);
                  RC_ASSERT(s_abc == s_cab);
              });
}

TEST(PredicatesPBT, Orient2DCoincident) {
    rc::check("orient2d with coincident points is Collinear",
              [](const Point2D& a, const Point2D& b) {
                  RC_PRE(std::isfinite(a.x) && std::isfinite(a.y));
                  RC_PRE(std::isfinite(b.x) && std::isfinite(b.y));

                  RC_ASSERT(orient2d(a, a, b) == Orientation::Collinear);
                  RC_ASSERT(orient2d(a, b, a) == Orientation::Collinear);
                  RC_ASSERT(orient2d(b, a, a) == Orientation::Collinear);
              });
}

TEST(PredicatesPBT, Orient2DPositiveScaling) {
    rc::check("orient2d sign is invariant under positive uniform scaling",
              [](const Point2D& a, const Point2D& b, const Point2D& c, int scale_int) {
                  RC_PRE(std::isfinite(a.x) && std::isfinite(a.y));
                  RC_PRE(std::isfinite(b.x) && std::isfinite(b.y));
                  RC_PRE(std::isfinite(c.x) && std::isfinite(c.y));

                  int factor = (std::abs(scale_int) % 50) + 1; // 1 to 50
                  double scale = static_cast<double>(factor);

                  Point2D sa{a.x * scale, a.y * scale};
                  Point2D sb{b.x * scale, b.y * scale};
                  Point2D sc{c.x * scale, c.y * scale};

                  int orig_s = sign_to_int(orient2d(a, b, c));
                  int scaled_s = sign_to_int(orient2d(sa, sb, sc));

                  RC_ASSERT(orig_s == scaled_s);
              });
}

TEST(PredicatesPBT, KnuthInteriorityAxiom) {
    rc::check("Knuth Axiom 4: (tbc > 0 && atc > 0 && abt > 0) ==> abc > 0",
              [](const Point2D& a, const Point2D& b, const Point2D& c, const Point2D& t) {
                  RC_PRE(std::isfinite(a.x) && std::isfinite(a.y));
                  RC_PRE(std::isfinite(b.x) && std::isfinite(b.y));
                  RC_PRE(std::isfinite(c.x) && std::isfinite(c.y));
                  RC_PRE(std::isfinite(t.x) && std::isfinite(t.y));

                  bool tbc = (orient2d(t, b, c) == Orientation::CounterClockwise);
                  bool atc = (orient2d(a, t, c) == Orientation::CounterClockwise);
                  bool abt = (orient2d(a, b, t) == Orientation::CounterClockwise);

                  if (tbc && atc && abt) {
                      RC_ASSERT(orient2d(a, b, c) == Orientation::CounterClockwise);
                  }
              });
}

TEST(PredicatesPBT, Orient3DAlternating) {
    rc::check("orient3d satisfies transposition antisymmetry",
              [](const Point3D& a, const Point3D& b, const Point3D& c, const Point3D& d) {
                  RC_PRE(std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z));
                  RC_PRE(std::isfinite(b.x) && std::isfinite(b.y) && std::isfinite(b.z));
                  RC_PRE(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z));
                  RC_PRE(std::isfinite(d.x) && std::isfinite(d.y) && std::isfinite(d.z));

                  int s_abcd = sign_to_int(orient3d(a, b, c, d));
                  int s_bacd = sign_to_int(orient3d(b, a, c, d));
                  int s_acbd = sign_to_int(orient3d(a, c, b, d));
                  int s_abdc = sign_to_int(orient3d(a, b, d, c));

                  RC_ASSERT(s_abcd == -s_bacd);
                  RC_ASSERT(s_abcd == -s_acbd);
                  RC_ASSERT(s_abcd == -s_abdc);
              });
}

TEST(PredicatesPBT, IncirclePermutationParity) {
    rc::check("incircle satisfies odd-permutation negation and even-permutation preservation",
              [](const Point2D& a, const Point2D& b, const Point2D& c, const Point2D& d) {
                  RC_PRE(std::isfinite(a.x) && std::isfinite(a.y));
                  RC_PRE(std::isfinite(b.x) && std::isfinite(b.y));
                  RC_PRE(std::isfinite(c.x) && std::isfinite(c.y));
                  RC_PRE(std::isfinite(d.x) && std::isfinite(d.y));

                  int s_abcd = sign_to_int(incircle(a, b, c, d));
                  int s_bacd = sign_to_int(incircle(b, a, c, d)); // 1 transposition -> -
                  int s_bcda = sign_to_int(incircle(b, c, d, a)); // 3 transpositions -> -
                  int s_cdab = sign_to_int(incircle(c, d, a, b)); // 2 transpositions -> +

                  RC_ASSERT(s_abcd == -s_bacd);
                  RC_ASSERT(s_abcd == -s_bcda);
                  RC_ASSERT(s_abcd == s_cdab);
              });
}

TEST(PredicatesPBT, InspherePermutationParity) {
    rc::check("insphere satisfies transposition negation and 5-cycle preservation",
              [](const Point3D& a, const Point3D& b, const Point3D& c, const Point3D& d,
                 const Point3D& e) {
                  RC_PRE(std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z));
                  RC_PRE(std::isfinite(b.x) && std::isfinite(b.y) && std::isfinite(b.z));
                  RC_PRE(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z));
                  RC_PRE(std::isfinite(d.x) && std::isfinite(d.y) && std::isfinite(d.z));
                  RC_PRE(std::isfinite(e.x) && std::isfinite(e.y) && std::isfinite(e.z));

                  int s_abcde = sign_to_int(insphere(a, b, c, d, e));
                  int s_bacde = sign_to_int(insphere(b, a, c, d, e)); // 1 transposition -> -
                  int s_bcdea = sign_to_int(insphere(b, c, d, e, a)); // 4 transpositions -> +

                  RC_ASSERT(s_abcde == -s_bacde);
                  RC_ASSERT(s_abcde == s_bcdea);
              });
}

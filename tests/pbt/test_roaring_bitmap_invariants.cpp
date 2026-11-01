#include "algoat/succinct/roaring_bitmap.hpp"

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <unordered_set>
#include <vector>

using namespace algoat::succinct;

namespace {

bool are_bitmaps_equal(const RoaringBitmap& a, const RoaringBitmap& b) {
    if (a.cardinality() != b.cardinality())
        return false;
    for (uint32_t val : a) {
        if (!b.contains(val))
            return false;
    }
    return true;
}

RoaringBitmap make_bitmap(const std::vector<uint32_t>& vals) {
    RoaringBitmap bm;
    for (uint32_t v : vals) {
        bm.add(v);
    }
    return bm;
}

} // namespace

// ============================================================================
// 1. Commutativity of Set Operations
// ============================================================================

RC_GTEST_PROP(RoaringBitmapPBT, IntersectionCommutativity,
              (const std::vector<uint32_t>& vals_a, const std::vector<uint32_t>& vals_b)) {
    RoaringBitmap a = make_bitmap(vals_a);
    RoaringBitmap b = make_bitmap(vals_b);

    RoaringBitmap ab = a & b;
    RoaringBitmap ba = b & a;

    RC_ASSERT(are_bitmaps_equal(ab, ba));
}

RC_GTEST_PROP(RoaringBitmapPBT, UnionCommutativity,
              (const std::vector<uint32_t>& vals_a, const std::vector<uint32_t>& vals_b)) {
    RoaringBitmap a = make_bitmap(vals_a);
    RoaringBitmap b = make_bitmap(vals_b);

    RoaringBitmap ab = a | b;
    RoaringBitmap ba = b | a;

    RC_ASSERT(are_bitmaps_equal(ab, ba));
}

RC_GTEST_PROP(RoaringBitmapPBT, XorCommutativity,
              (const std::vector<uint32_t>& vals_a, const std::vector<uint32_t>& vals_b)) {
    RoaringBitmap a = make_bitmap(vals_a);
    RoaringBitmap b = make_bitmap(vals_b);

    RoaringBitmap ab = a ^ b;
    RoaringBitmap ba = b ^ a;

    RC_ASSERT(are_bitmaps_equal(ab, ba));
}

// ============================================================================
// 2. Identity and Set Invariants
// ============================================================================

RC_GTEST_PROP(RoaringBitmapPBT, IdentityLaws, (const std::vector<uint32_t>& vals)) {
    RoaringBitmap a = make_bitmap(vals);
    RoaringBitmap empty;

    RC_ASSERT(are_bitmaps_equal(a | empty, a));
    RC_ASSERT((a & empty).empty());
    RC_ASSERT((a - a).empty());
    RC_ASSERT(are_bitmaps_equal(a - empty, a));
    RC_ASSERT((a ^ a).empty());
}

RC_GTEST_PROP(RoaringBitmapPBT, InclusionExclusionLaw,
              (const std::vector<uint32_t>& vals_a, const std::vector<uint32_t>& vals_b)) {
    RoaringBitmap a = make_bitmap(vals_a);
    RoaringBitmap b = make_bitmap(vals_b);

    RoaringBitmap un = a | b;
    RoaringBitmap inter = a & b;

    RC_ASSERT(un.cardinality() == a.cardinality() + b.cardinality() - inter.cardinality());
}

// ============================================================================
// 3. Jaccard Index Invariants
// ============================================================================

RC_GTEST_PROP(RoaringBitmapPBT, JaccardBoundsAndSymmetry,
              (const std::vector<uint32_t>& vals_a, const std::vector<uint32_t>& vals_b)) {
    RoaringBitmap a = make_bitmap(vals_a);
    RoaringBitmap b = make_bitmap(vals_b);

    double jab = RoaringBitmap::jaccard_index(a, b);
    double jba = RoaringBitmap::jaccard_index(b, a);

    RC_ASSERT(jab >= 0.0 && jab <= 1.0);
    RC_ASSERT(std::abs(jab - jba) < 1e-9);

    if (!a.empty()) {
        RC_ASSERT(std::abs(RoaringBitmap::jaccard_index(a, a) - 1.0) < 1e-9);
    }
}

// ============================================================================
// 4. Serialization Round-Trip Idempotence
// ============================================================================

RC_GTEST_PROP(RoaringBitmapPBT, SerializationRoundTrip, (const std::vector<uint32_t>& vals)) {
    RoaringBitmap a = make_bitmap(vals);

    auto bytes = a.serialize();
    RoaringBitmap restored = RoaringBitmap::deserialize(bytes);

    RC_ASSERT(are_bitmaps_equal(a, restored));
}

// ============================================================================
// 5. Run Optimization Equivalence
// ============================================================================

RC_GTEST_PROP(RoaringBitmapPBT, RunOptimizationPreservesEquivalence,
              (const std::vector<uint32_t>& vals)) {
    RoaringBitmap a = make_bitmap(vals);
    RoaringBitmap opt = a;
    opt.run_optimize();

    RC_ASSERT(are_bitmaps_equal(a, opt));

    auto opt_bytes = opt.serialize();
    RoaringBitmap deser = RoaringBitmap::deserialize(opt_bytes);
    RC_ASSERT(are_bitmaps_equal(a, deser));
}

// ============================================================================
// 6. Differential Equivalence vs std::unordered_set
// ============================================================================

RC_GTEST_PROP(RoaringBitmapPBT, DifferentialTestingAgainstStdSet,
              (const std::vector<uint32_t>& to_add, const std::vector<uint32_t>& to_remove)) {
    RoaringBitmap bm;
    std::unordered_set<uint32_t> ref;

    for (uint32_t v : to_add) {
        bool bm_res = bm.add(v);
        bool ref_res = ref.insert(v).second;
        RC_ASSERT(bm_res == ref_res);
    }

    RC_ASSERT(bm.cardinality() == ref.size());

    for (uint32_t v : to_remove) {
        bool bm_res = bm.remove(v);
        bool ref_res = (ref.erase(v) > 0);
        RC_ASSERT(bm_res == ref_res);
    }

    RC_ASSERT(bm.cardinality() == ref.size());

    for (uint32_t v : ref) {
        RC_ASSERT(bm.contains(v));
    }
}

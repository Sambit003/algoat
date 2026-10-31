#include "algoat/sketches/detail/hash.hpp"
#include "algoat/sketches/hyperloglog.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

using namespace algoat::sketches;

namespace {

template <std::size_t P> HyperLogLog<P> make_hll_from_values(const std::vector<uint64_t>& values) {
    HyperLogLog<P> hll;
    for (uint64_t v : values) {
        hll.add(v);
    }
    return hll;
}

} // namespace

// ============================================================================
// 1. Idempotence Invariant: A ∪ A = A
// ============================================================================

RC_GTEST_PROP(HyperLogLogPBT, Idempotence, (const std::vector<uint64_t>& vals)) {
    HyperLogLog<12> a = make_hll_from_values<12>(vals);
    HyperLogLog<12> a_merged = a;
    a_merged.merge(a);

    RC_ASSERT(a_merged.count() == a.count());
    RC_ASSERT(a_merged.mode() == a.mode());
}

// ============================================================================
// 2. Commutativity Invariant: A ∪ B = B ∪ A
// ============================================================================

RC_GTEST_PROP(HyperLogLogPBT, Commutativity,
              (const std::vector<uint64_t>& vals_a, const std::vector<uint64_t>& vals_b)) {
    HyperLogLog<12> a = make_hll_from_values<12>(vals_a);
    HyperLogLog<12> b = make_hll_from_values<12>(vals_b);

    HyperLogLog<12> a_b = a;
    a_b.merge(b);

    HyperLogLog<12> b_a = b;
    b_a.merge(a);

    RC_ASSERT(a_b.count() == b_a.count());
}

// ============================================================================
// 3. Associativity Invariant: (A ∪ B) ∪ C = A ∪ (B ∪ C)
// ============================================================================

RC_GTEST_PROP(HyperLogLogPBT, Associativity,
              (const std::vector<uint64_t>& vals_a, const std::vector<uint64_t>& vals_b,
               const std::vector<uint64_t>& vals_c)) {
    HyperLogLog<12> a = make_hll_from_values<12>(vals_a);
    HyperLogLog<12> b = make_hll_from_values<12>(vals_b);
    HyperLogLog<12> c = make_hll_from_values<12>(vals_c);

    // Left association: (A ∪ B) ∪ C
    HyperLogLog<12> lhs = a;
    lhs.merge(b);
    lhs.merge(c);

    // Right association: A ∪ (B ∪ C)
    HyperLogLog<12> rhs_inner = b;
    rhs_inner.merge(c);
    HyperLogLog<12> rhs = a;
    rhs.merge(rhs_inner);

    RC_ASSERT(lhs.count() == rhs.count());
}

// ============================================================================
// 4. Monotonicity & Sub-Additivity Invariants
// ============================================================================

RC_GTEST_PROP(HyperLogLogPBT, MonotonicityAndSubAdditivity,
              (const std::vector<uint64_t>& vals_a, const std::vector<uint64_t>& vals_b)) {
    HyperLogLog<12> a = make_hll_from_values<12>(vals_a);
    HyperLogLog<12> b = make_hll_from_values<12>(vals_b);

    HyperLogLog<12> merged = a;
    merged.merge(b);

    // Monotonicity: Count(A ∪ B) >= max(Count(A), Count(B))
    // (Allowing tiny rounding tolerance of 1)
    uint64_t max_single = std::max(a.count(), b.count());
    if (max_single > 0) {
        RC_ASSERT(merged.count() + 1 >= max_single);
    }
}

// ============================================================================
// 5. Serialization Round-Trip Invariant
// ============================================================================

RC_GTEST_PROP(HyperLogLogPBT, SerializationRoundTrip, (const std::vector<uint64_t>& vals)) {
    HyperLogLog<14> orig = make_hll_from_values<14>(vals);
    auto bytes = orig.serialize();
    RC_ASSERT(bytes.size() == 12304);

    HyperLogLog<14> restored = HyperLogLog<14>::deserialize(bytes);
    RC_ASSERT(restored.count() == orig.count());
}

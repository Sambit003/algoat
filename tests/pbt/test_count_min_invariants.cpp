#include "algoat/sketches/count_min.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <unordered_map>
#include <vector>

using namespace algoat::sketches;

// ============================================================================
// 1. CountMinSketch: Non-Underestimation Invariant
// ============================================================================

RC_GTEST_PROP(CountMinSketchPBT, NonUnderestimation, (const std::vector<uint64_t>& keys)) {
    CountMinSketch<uint64_t, uint32_t> cms(0.01, 0.05, true);
    std::unordered_map<uint64_t, uint32_t> exact;

    for (uint64_t k : keys) {
        cms.add(k, 1);
        exact[k] += 1;
    }

    for (const auto& [k, true_f] : exact) {
        RC_ASSERT(cms.estimate(k) >= true_f);
    }
}

// ============================================================================
// 2. CountMinSketch: Standard Merge Commutativity
// ============================================================================

RC_GTEST_PROP(CountMinSketchPBT, MergeCommutativity,
              (const std::vector<uint64_t>& keys_a, const std::vector<uint64_t>& keys_b)) {
    CountMinSketch<uint64_t, uint32_t> a(0.01, 0.05, false);
    CountMinSketch<uint64_t, uint32_t> b(0.01, 0.05, false);

    for (uint64_t k : keys_a) {
        a.add(k, 1);
    }
    for (uint64_t k : keys_b) {
        b.add(k, 1);
    }

    CountMinSketch<uint64_t, uint32_t> a_b = a;
    a_b.merge(b);

    CountMinSketch<uint64_t, uint32_t> b_a = b;
    b_a.merge(a);

    for (uint64_t k : keys_a) {
        RC_ASSERT(a_b.estimate(k) == b_a.estimate(k));
    }
    for (uint64_t k : keys_b) {
        RC_ASSERT(a_b.estimate(k) == b_a.estimate(k));
    }
}

// ============================================================================
// 3. CountMinSketch: Standard Merge Associativity
// ============================================================================

RC_GTEST_PROP(CountMinSketchPBT, MergeAssociativity,
              (const std::vector<uint64_t>& keys_a, const std::vector<uint64_t>& keys_b,
               const std::vector<uint64_t>& keys_c)) {
    CountMinSketch<uint64_t, uint32_t> a(0.01, 0.05, false);
    CountMinSketch<uint64_t, uint32_t> b(0.01, 0.05, false);
    CountMinSketch<uint64_t, uint32_t> c(0.01, 0.05, false);

    for (uint64_t k : keys_a)
        a.add(k, 1);
    for (uint64_t k : keys_b)
        b.add(k, 1);
    for (uint64_t k : keys_c)
        c.add(k, 1);

    // (A ∪ B) ∪ C
    CountMinSketch<uint64_t, uint32_t> ab_c = a;
    ab_c.merge(b);
    ab_c.merge(c);

    // A ∪ (B ∪ C)
    CountMinSketch<uint64_t, uint32_t> bc = b;
    bc.merge(c);
    CountMinSketch<uint64_t, uint32_t> a_bc = a;
    a_bc.merge(bc);

    for (uint64_t k : keys_a) {
        RC_ASSERT(ab_c.estimate(k) == a_bc.estimate(k));
    }
    for (uint64_t k : keys_b) {
        RC_ASSERT(ab_c.estimate(k) == a_bc.estimate(k));
    }
    for (uint64_t k : keys_c) {
        RC_ASSERT(ab_c.estimate(k) == a_bc.estimate(k));
    }
}

// ============================================================================
// 4. SpaceSaving: Conservation of Total Stream Weight
// ============================================================================

RC_GTEST_PROP(SpaceSavingPBT, ConservationOfStreamWeight, (const std::vector<uint64_t>& keys)) {
    RC_PRE(!keys.empty());

    SpaceSavingHeavyHitters<uint64_t, 16> ss;
    for (uint64_t k : keys) {
        ss.add(k, 1);
    }

    auto items = ss.top_k();
    uint64_t total_count = 0;
    for (const auto& item : items) {
        total_count += item.count;
    }

    RC_ASSERT(total_count == static_cast<uint64_t>(keys.size()));
}

// ============================================================================
// 5. SpaceSaving: Fundamental Error Bound Invariant
// ============================================================================

RC_GTEST_PROP(SpaceSavingPBT, ErrorBoundGuarantee, (const std::vector<uint64_t>& keys)) {
    RC_PRE(!keys.empty());

    SpaceSavingHeavyHitters<uint64_t, 16> ss;
    std::unordered_map<uint64_t, uint64_t> exact;

    for (uint64_t k : keys) {
        ss.add(k, 1);
        exact[k] += 1;
    }

    auto items = ss.top_k();
    for (const auto& item : items) {
        uint64_t true_f = exact[item.key];
        RC_ASSERT(item.count >= true_f);
        RC_ASSERT(item.count - item.error <= true_f);
    }
}

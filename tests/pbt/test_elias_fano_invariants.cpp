#include "algoat/succinct/elias_fano.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

using namespace algoat::succinct;

// ============================================================================
// 1. Invariant: Round-Trip Equality at(i) == input[i]
// ============================================================================

RC_GTEST_PROP(EliasFanoPBT, RoundTripEquality, (std::vector<uint32_t> raw_nums)) {
    if (raw_nums.empty()) {
        EliasFanoEncoder enc(0, 100);
        EliasFanoList list(enc.finalize());
        RC_ASSERT(list.size() == 0ULL);
        return;
    }

    std::vector<uint64_t> seq;
    seq.reserve(raw_nums.size());
    for (uint32_t x : raw_nums) {
        seq.push_back(static_cast<uint64_t>(x));
    }
    std::sort(seq.begin(), seq.end());

    uint64_t universe = seq.back() + 10ULL;
    EliasFanoEncoder encoder(seq.size(), universe);
    for (uint64_t x : seq) {
        encoder.push(x);
    }
    std::vector<uint8_t> buf = encoder.finalize();
    EliasFanoList list(buf);

    RC_ASSERT(list.size() == seq.size());
    RC_ASSERT(list.universe() == universe);

    for (size_t i = 0; i < seq.size(); ++i) {
        RC_ASSERT(list.at(i) == seq[i]);
    }
}

// ============================================================================
// 2. Invariant: Monotonicity at(i) <= at(i+1)
// ============================================================================

RC_GTEST_PROP(EliasFanoPBT, MonotonicityInvariant, (std::vector<uint32_t> raw_nums)) {
    if (raw_nums.size() < 2)
        return;

    std::vector<uint64_t> seq;
    seq.reserve(raw_nums.size());
    for (uint32_t x : raw_nums) {
        seq.push_back(static_cast<uint64_t>(x));
    }
    std::sort(seq.begin(), seq.end());

    uint64_t universe = seq.back() + 1ULL;
    EliasFanoEncoder encoder(seq.size(), universe);
    for (uint64_t x : seq) {
        encoder.push(x);
    }
    EliasFanoList list(encoder.finalize());

    for (size_t i = 0; i + 1 < seq.size(); ++i) {
        RC_ASSERT(list.at(i) <= list.at(i + 1));
    }
}

// ============================================================================
// 3. Invariant: Successor Search Correctness (next_geq and lower_bound_pos)
// ============================================================================

RC_GTEST_PROP(EliasFanoPBT, SuccessorSearchInvariant,
              (std::vector<uint32_t> raw_nums, uint32_t query_raw)) {
    if (raw_nums.empty())
        return;

    std::vector<uint64_t> seq;
    seq.reserve(raw_nums.size());
    for (uint32_t x : raw_nums) {
        seq.push_back(static_cast<uint64_t>(x));
    }
    std::sort(seq.begin(), seq.end());

    uint64_t universe = seq.back() + 50ULL;
    EliasFanoEncoder encoder(seq.size(), universe);
    for (uint64_t x : seq) {
        encoder.push(x);
    }
    EliasFanoList list(encoder.finalize());

    uint64_t target = static_cast<uint64_t>(query_raw % (universe + 10ULL));

    size_t actual_pos = list.lower_bound_pos(target);
    uint64_t actual_val = list.next_geq(target);

    auto std_it = std::lower_bound(seq.begin(), seq.end(), target);
    size_t expected_pos = static_cast<size_t>(std::distance(seq.begin(), std_it));
    uint64_t expected_val = (std_it != seq.end()) ? *std_it : universe;

    RC_ASSERT(actual_pos == expected_pos);
    RC_ASSERT(actual_val == expected_val);
}

// ============================================================================
// 4. Invariant: Sequential Iterator Equivalence
// ============================================================================

RC_GTEST_PROP(EliasFanoPBT, IteratorEquivalence, (std::vector<uint32_t> raw_nums)) {
    std::vector<uint64_t> seq;
    seq.reserve(raw_nums.size());
    for (uint32_t x : raw_nums) {
        seq.push_back(static_cast<uint64_t>(x));
    }
    std::sort(seq.begin(), seq.end());

    uint64_t universe = seq.empty() ? 100ULL : (seq.back() + 10ULL);
    EliasFanoEncoder encoder(seq.size(), universe);
    for (uint64_t x : seq) {
        encoder.push(x);
    }
    EliasFanoList list(encoder.finalize());

    std::vector<uint64_t> iterated;
    for (uint64_t val : list) {
        iterated.push_back(val);
    }

    RC_ASSERT(iterated == seq);
}

// ============================================================================
// 5. Invariant: Zero-Copy Borrowing Equivalence
// ============================================================================

RC_GTEST_PROP(EliasFanoPBT, ZeroCopySpanEquivalence, (std::vector<uint32_t> raw_nums)) {
    if (raw_nums.empty())
        return;

    std::vector<uint64_t> seq;
    seq.reserve(raw_nums.size());
    for (uint32_t x : raw_nums) {
        seq.push_back(static_cast<uint64_t>(x));
    }
    std::sort(seq.begin(), seq.end());

    uint64_t universe = seq.back() + 1ULL;
    EliasFanoEncoder encoder(seq.size(), universe);
    for (uint64_t x : seq) {
        encoder.push(x);
    }
    std::vector<uint8_t> buf = encoder.finalize();

    EliasFanoList owning_list(buf);
    EliasFanoList borrowing_list(std::span<const uint8_t>{buf.data(), buf.size()});

    RC_ASSERT(borrowing_list.size() == owning_list.size());
    RC_ASSERT(borrowing_list.universe() == owning_list.universe());

    for (size_t i = 0; i < seq.size(); ++i) {
        RC_ASSERT(borrowing_list.at(i) == owning_list.at(i));
    }
}

#include "algoat/succinct/wavelet_matrix.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

using namespace algoat::succinct;

// ============================================================================
// 1. Invariant: Access Round-Trip
// ============================================================================

RC_GTEST_PROP(WaveletMatrixPBT, AccessRoundTrip, (std::vector<uint16_t> input)) {
    WaveletMatrix<uint16_t> wm(input);
    RC_ASSERT(wm.size() == input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        RC_ASSERT(wm.access(i) == input[i]);
    }
}

// ============================================================================
// 2. Invariant: Select-Access Invertibility
// ============================================================================

RC_GTEST_PROP(WaveletMatrixPBT, SelectAccessInvertibility, (std::vector<uint8_t> input)) {
    if (input.empty()) {
        return;
    }

    WaveletMatrix<uint8_t> wm(input);

    for (uint32_t c = 0; c < 256; ++c) {
        uint8_t val = static_cast<uint8_t>(c);
        size_t total = wm.rank(val, input.size());
        for (size_t k = 1; k <= total; ++k) {
            size_t pos = wm.select(val, k);
            RC_ASSERT(pos < input.size());
            RC_ASSERT(wm.access(pos) == val);
        }
        // k out of bounds returns size
        RC_ASSERT(wm.select(val, total + 1) == input.size());
        RC_ASSERT(wm.select(val, 0) == input.size());
    }
}

// ============================================================================
// 3. Invariant: Select-Rank Correspondence
// ============================================================================

RC_GTEST_PROP(WaveletMatrixPBT, SelectRankCorrespondence, (std::vector<uint16_t> input)) {
    if (input.empty()) {
        return;
    }

    WaveletMatrix<uint16_t> wm(input);

    for (uint16_t val : input) {
        size_t total = wm.rank(val, input.size());
        for (size_t k = 1; k <= total; ++k) {
            size_t pos = wm.select(val, k);
            RC_ASSERT(wm.rank(val, pos + 1) == k);
            RC_ASSERT(wm.rank(val, pos) == k - 1);
        }
    }
}

// ============================================================================
// 4. Invariant: Range Quantile Correctness vs nth_element
// ============================================================================

RC_GTEST_PROP(WaveletMatrixPBT, RangeQuantileCorrectness, (std::vector<uint32_t> input)) {
    if (input.empty()) {
        return;
    }

    WaveletMatrix<uint32_t> wm(input);
    const size_t n = input.size();

    auto l = *rc::gen::inRange<size_t>(0, n);
    auto r = *rc::gen::inRange<size_t>(l + 1, n + 1);
    auto k = *rc::gen::inRange<size_t>(0, r - l);

    std::vector<uint32_t> sub(input.begin() + l, input.begin() + r);
    std::nth_element(sub.begin(), sub.begin() + k, sub.end());
    uint32_t expected = sub[k];

    RC_ASSERT(wm.range_quantile(l, r, k) == expected);
}

// ============================================================================
// 5. Invariant: Range Count Additivity & Rank Difference
// ============================================================================

RC_GTEST_PROP(WaveletMatrixPBT, RangeCountAdditivityAndRank, (std::vector<uint16_t> input)) {
    if (input.empty()) {
        return;
    }

    WaveletMatrix<uint16_t> wm(input);
    const size_t n = input.size();

    auto l = *rc::gen::inRange<size_t>(0, n);
    auto r = *rc::gen::inRange<size_t>(l + 1, n + 1);

    auto val_a = *rc::gen::inRange<uint16_t>(0, 100);
    auto val_b = *rc::gen::inRange<uint16_t>(val_a, 100);
    auto val_c = *rc::gen::inRange<uint16_t>(val_b, 100);

    // Range count single value matches rank difference
    RC_ASSERT(wm.range_count(l, r, val_a, val_a) == (wm.rank(val_a, r) - wm.rank(val_a, l)));

    // Additivity: count[a..b] + count[b+1..c] == count[a..c]
    if (val_b < val_c) {
        size_t part1 = wm.range_count(l, r, val_a, val_b);
        size_t part2 = wm.range_count(l, r, static_cast<uint16_t>(val_b + 1), val_c);
        size_t whole = wm.range_count(l, r, val_a, val_c);
        RC_ASSERT(part1 + part2 == whole);
    }
}

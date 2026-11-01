#include "algoat/succinct/rank_select.hpp"

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

using namespace algoat::succinct;

// ============================================================================
// 1. Rank Partition Invariant: rank1(pos) + rank0(pos) == pos
// ============================================================================

RC_GTEST_PROP(RankSelectPBT, RankPartitionSumEqualsPos, (const std::vector<bool>& bits)) {
    RankSelectBitVector bv(bits);
    size_t n = bits.size();

    // Check boundary positions
    RC_ASSERT(bv.rank1(0) == 0ULL);
    RC_ASSERT(bv.rank0(0) == 0ULL);
    RC_ASSERT(bv.rank1(n) == bv.total_ones());
    RC_ASSERT(bv.rank0(n) == n - bv.total_ones());

    // Sample positions across the bitvector
    if (n > 0) {
        size_t step = std::max<size_t>(1, n / 20);
        for (size_t pos = 0; pos <= n; pos += step) {
            RC_ASSERT(bv.rank1(pos) + bv.rank0(pos) == pos);
        }
    }
}

// ============================================================================
// 2. Rank Monotonicity Invariant: pos1 <= pos2 => rank1(pos1) <= rank1(pos2)
// ============================================================================

RC_GTEST_PROP(RankSelectPBT, RankMonotonicity, (const std::vector<bool>& bits)) {
    RankSelectBitVector bv(bits);
    size_t n = bits.size();

    uint64_t prev_rank1 = 0;
    uint64_t prev_rank0 = 0;
    size_t step = std::max<size_t>(1, n / 30);
    for (size_t pos = 0; pos <= n; pos += step) {
        uint64_t r1 = bv.rank1(pos);
        uint64_t r0 = bv.rank0(pos);
        RC_ASSERT(r1 >= prev_rank1);
        RC_ASSERT(r0 >= prev_rank0);
        prev_rank1 = r1;
        prev_rank0 = r0;
    }
}

// ============================================================================
// 3. Select1 Inversion Invariant: rank1(select1(k)) == k - 1 && access(select1(k)) == true
// ============================================================================

RC_GTEST_PROP(RankSelectPBT, Select1Inversion, (const std::vector<bool>& bits)) {
    RankSelectBitVector bv(bits);
    uint64_t ones = bv.total_ones();

    size_t prev_pos = 0;
    for (uint64_t k = 1; k <= ones; ++k) {
        size_t pos = bv.select1(k);
        RC_ASSERT(pos < bits.size());
        RC_ASSERT(bv.access(pos) == true);
        RC_ASSERT(bv.rank1(pos) == k - 1);
        if (k > 1) {
            RC_ASSERT(pos > prev_pos);
        }
        prev_pos = pos;
    }
}

// ============================================================================
// 4. Select0 Inversion Invariant: rank0(select0(k)) == k - 1 && access(select0(k)) == false
// ============================================================================

RC_GTEST_PROP(RankSelectPBT, Select0Inversion, (const std::vector<bool>& bits)) {
    RankSelectBitVector bv(bits);
    uint64_t zeros = bv.num_bits() - bv.total_ones();

    size_t prev_pos = 0;
    for (uint64_t k = 1; k <= zeros; ++k) {
        size_t pos = bv.select0(k);
        RC_ASSERT(pos < bits.size());
        RC_ASSERT(bv.access(pos) == false);
        RC_ASSERT(bv.rank0(pos) == k - 1);
        if (k > 1) {
            RC_ASSERT(pos > prev_pos);
        }
        prev_pos = pos;
    }
}

// ============================================================================
// 5. Packed Words vs Span of Bool Equivalence
// ============================================================================

RC_GTEST_PROP(RankSelectPBT, PackedWordsEquivalence, (const std::vector<uint64_t>& words)) {
    if (words.empty()) {
        return;
    }
    size_t n = words.size() * 64;
    std::vector<bool> bits(n);
    for (size_t i = 0; i < n; ++i) {
        bits[i] = ((words[i / 64] >> (i % 64)) & 1ULL) != 0ULL;
    }

    RankSelectBitVector bv_words(std::span<const uint64_t>{words}, n);
    RankSelectBitVector bv_bits(bits);

    RC_ASSERT(bv_words.num_bits() == bv_bits.num_bits());
    RC_ASSERT(bv_words.total_ones() == bv_bits.total_ones());

    size_t step = std::max<size_t>(1, n / 15);
    for (size_t pos = 0; pos <= n; pos += step) {
        RC_ASSERT(bv_words.rank1(pos) == bv_bits.rank1(pos));
        RC_ASSERT(bv_words.rank0(pos) == bv_bits.rank0(pos));
    }
}

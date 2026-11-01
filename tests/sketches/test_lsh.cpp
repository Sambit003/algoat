#include "algoat/sketches/lsh.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <span>
#include <vector>

using namespace algoat::sketches;

// ============================================================================
// Slice 1: BBitMinHash Concepts, Layout, Alignment, and Size Invariants
// ============================================================================

TEST(BBitMinHashTest, ConceptAndSizeInvariants) {
    // Check concept requirements:
    // B must be 1, 2, or 4 and K must be divisible by (64 / B)
    static_assert(requires { typename BBitMinHash<256, 2>; });
    static_assert(requires { typename BBitMinHash<64, 1>; });
    static_assert(requires { typename BBitMinHash<512, 2>; });
    static_assert(requires { typename BBitMinHash<128, 4>; });

    // Acceptance Criterion 3:
    // Signature size for K=256, B=2 is exactly 64 bytes (sizeof(Signature) == 64),
    // matching one CPU cache line.
    using MinHash256_2 = BBitMinHash<256, 2>;
    static_assert(MinHash256_2::kNumWords == 8);
    static_assert(sizeof(MinHash256_2::Signature) == 64,
                  "Signature size for K=256, B=2 must be exactly 64 bytes (one CPU cache line)");
    static_assert(alignof(MinHash256_2::Signature) == 64,
                  "Signature must be aligned to 64 bytes (alignas(64))");

    // Acceptance Criterion 4:
    // Zero-Allocation Execution: Signature distance and similarity methods are completely noexcept
    MinHash256_2::Signature s1{};
    MinHash256_2::Signature s2{};
    static_assert(noexcept(s1.jaccard_similarity(s2)));
    static_assert(noexcept(s1.hamming_distance(s2)));
    static_assert(noexcept(MinHash256_2::estimate_jaccard(0.5)));
}

// ============================================================================
// Slice 2: BBitMinHash Similarity and Statistical Accuracy
// ============================================================================

TEST(BBitMinHashTest, ExactMatchesAndSelfSimilarity) {
    BBitMinHash<256, 2> minhash(42);

    std::vector<uint64_t> tokens = {10, 20, 30, 40, 50, 60, 70, 80};
    auto sig1 = minhash.compute_signature(tokens);
    auto sig2 = minhash.compute_signature(tokens);

    EXPECT_EQ(sig1.hamming_distance(sig2), 0U);
    EXPECT_DOUBLE_EQ(sig1.jaccard_similarity(sig2), 1.0);
}

TEST(BBitMinHashTest, DisjointSetsLowSimilarity) {
    BBitMinHash<512, 2> minhash(12345);

    // Two disjoint large sets
    std::vector<uint64_t> set_a(1000);
    std::vector<uint64_t> set_b(1000);
    for (size_t i = 0; i < 1000; ++i) {
        set_a[i] = i * 2;
        set_b[i] = i * 2 + 1;
    }

    auto sig_a = minhash.compute_signature(set_a);
    auto sig_b = minhash.compute_signature(set_b);

    double est_jaccard = sig_a.jaccard_similarity(sig_b);
    EXPECT_GE(est_jaccard, 0.0);
    EXPECT_LE(est_jaccard, 0.15); // Disjoint sets should estimate near 0
}

TEST(BBitMinHashTest, JaccardEstimationPrecisionRMSE) {
    // Acceptance Criterion 1:
    // On synthetic pairs with true J in [0.1, 0.9], b-Bit MinHash (K=512, b=2)
    // achieves root-mean-square error (RMSE) within sigma <= 0.035 of theoretical bounds.
    BBitMinHash<512, 2> minhash(987654321ULL);

    const std::vector<double> target_jaccards = {0.1, 0.25, 0.5, 0.75, 0.9};
    constexpr size_t kTrialsPerJ = 25;

    double sum_squared_error = 0.0;
    size_t total_evals = 0;

    for (double target_j : target_jaccards) {
        constexpr size_t common_size = 1000;
        // J = C / (C + 2*D) => D = round(C * (1 - J) / (2 * J))
        const size_t diff_size = static_cast<size_t>(
            std::round(static_cast<double>(common_size) * (1.0 - target_j) / (2.0 * target_j)));

        const double actual_jaccard =
            static_cast<double>(common_size) / static_cast<double>(common_size + 2 * diff_size);

        for (size_t trial = 0; trial < kTrialsPerJ; ++trial) {
            uint64_t offset = (trial + 1) * 100000ULL;
            std::vector<uint64_t> a;
            std::vector<uint64_t> b;
            a.reserve(common_size + diff_size);
            b.reserve(common_size + diff_size);

            for (size_t i = 0; i < common_size; ++i) {
                uint64_t elem = offset + i;
                a.push_back(elem);
                b.push_back(elem);
            }
            for (size_t i = 0; i < diff_size; ++i) {
                a.push_back(offset + common_size + i);
            }
            for (size_t i = 0; i < diff_size; ++i) {
                b.push_back(offset + common_size + diff_size + i);
            }

            auto sig_a = minhash.compute_signature(a);
            auto sig_b = minhash.compute_signature(b);
            double estimated_j = sig_a.jaccard_similarity(sig_b);

            double diff = estimated_j - actual_jaccard;
            sum_squared_error += diff * diff;
            ++total_evals;
        }
    }

    double rmse = std::sqrt(sum_squared_error / static_cast<double>(total_evals));
    EXPECT_LE(rmse, 0.035);
}

// ============================================================================
// Slice 3: SimHash Fingerprint, Cosine Similarity, and Hamming Distance
// ============================================================================

TEST(SimHashTest, EmptyInputProducesZero) {
    EXPECT_EQ(SimHash::compute_64({}), 0ULL);
}

TEST(SimHashTest, IdenticalInputsProduceZeroHammingDistance) {
    std::vector<uint64_t> tokens = {0x1111222233334444ULL, 0x5555666677778888ULL,
                                    0x9999AAAABBBBCCCCULL};
    uint64_t fp1 = SimHash::compute_64(tokens);
    uint64_t fp2 = SimHash::compute_64(tokens);

    EXPECT_EQ(SimHash::hamming_distance(fp1, fp2), 0U);
    EXPECT_NEAR(SimHash::cosine_similarity(fp1, fp2), 1.0, 1e-6);
}

TEST(SimHashTest, CosineDistanceCharikarFormula) {
    // Test theoretical cosine formula against known Hamming distances:
    // d_H = 0 -> cos(0) = 1.0
    EXPECT_NEAR(SimHash::cosine_similarity(0ULL, 0ULL), 1.0, 1e-6);

    // d_H = 32 -> cos(pi * 32 / 64) = cos(pi / 2) = 0.0 (orthogonal)
    uint64_t fp_a = 0x00000000FFFFFFFFULL;
    uint64_t fp_b = 0x0000000000000000ULL;
    EXPECT_EQ(SimHash::hamming_distance(fp_a, fp_b), 32U);
    EXPECT_NEAR(SimHash::cosine_similarity(fp_a, fp_b), 0.0, 1e-6);

    // d_H = 64 -> cos(pi) = -1.0 (diametrically opposed)
    uint64_t fp_inv = ~fp_b;
    EXPECT_EQ(SimHash::hamming_distance(fp_b, fp_inv), 64U);
    EXPECT_NEAR(SimHash::cosine_similarity(fp_b, fp_inv), -1.0, 1e-6);
}

TEST(SimHashTest, WeightedVsUnweightedEquivalence) {
    std::vector<uint64_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> unit_weights(tokens.size(), 1.0f);

    uint64_t fp_unweighted = SimHash::compute_64(tokens);
    uint64_t fp_weighted = SimHash::compute_64(tokens, unit_weights);

    EXPECT_EQ(fp_unweighted, fp_weighted);
}

TEST(SimHashTest, NegativeAndZeroWeights) {
    std::vector<uint64_t> tokens = {0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL};
    std::vector<float> weights_zero = {0.0f, 1.0f};

    uint64_t fp_partial = SimHash::compute_64(tokens, weights_zero);
    uint64_t fp_single = SimHash::compute_64(std::span<const uint64_t>{&tokens[1], 1});
    EXPECT_EQ(fp_partial, fp_single);

    // Negative weight should invert the direction
    std::vector<uint64_t> single_token = {0x00000000FFFFFFFFULL};
    std::vector<float> pos_w = {2.0f};
    std::vector<float> neg_w = {-2.0f};

    uint64_t fp_pos = SimHash::compute_64(single_token, pos_w);
    uint64_t fp_neg = SimHash::compute_64(single_token, neg_w);
    EXPECT_EQ(fp_pos ^ fp_neg, ~0ULL);
}

TEST(BBitMinHashTest, DifferentBitWidthsB1AndB4) {
    BBitMinHash<64, 1> minhash_b1(111);
    std::vector<uint64_t> doc_a = {1, 2, 3, 4, 5};
    std::vector<uint64_t> doc_b = {1, 2, 3, 4, 5};
    auto sig_a1 = minhash_b1.compute_signature(doc_a);
    auto sig_b1 = minhash_b1.compute_signature(doc_b);
    EXPECT_EQ(sig_a1.hamming_distance(sig_b1), 0U);
    EXPECT_DOUBLE_EQ(sig_a1.jaccard_similarity(sig_b1), 1.0);

    BBitMinHash<128, 4> minhash_b4(222);
    auto sig_a4 = minhash_b4.compute_signature(doc_a);
    auto sig_b4 = minhash_b4.compute_signature(doc_b);
    EXPECT_EQ(sig_a4.hamming_distance(sig_b4), 0U);
    EXPECT_DOUBLE_EQ(sig_a4.jaccard_similarity(sig_b4), 1.0);

    // Bounds clipping: estimate_jaccard must always be in [0.0, 1.0]
    EXPECT_GE(minhash_b1.estimate_jaccard(0.0), 0.0);
    EXPECT_LE(minhash_b1.estimate_jaccard(1.0), 1.0);
    EXPECT_GE(minhash_b4.estimate_jaccard(0.0), 0.0);
    EXPECT_LE(minhash_b4.estimate_jaccard(1.0), 1.0);
}

// ============================================================================
// Slice 4: Banded LSHIndex Candidate Retrieval and Deduplication
// ============================================================================

TEST(LSHIndexTest, ExactAndNearDuplicateCandidateRetrieval) {
    // 16 bands, 16 rows per band = 256 rows
    using MinHash = BBitMinHash<256, 2>;
    LSHIndex<MinHash::Signature, 16, 16> index;

    MinHash generator(777);

    // Item 1
    std::vector<uint64_t> doc1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto sig1 = generator.compute_signature(doc1);
    index.insert(1001, sig1);

    // Item 2 (near duplicate of 1)
    std::vector<uint64_t> doc2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 999};
    auto sig2 = generator.compute_signature(doc2);
    index.insert(1002, sig2);

    // Item 3 (completely different doc)
    std::vector<uint64_t> doc3 = {500, 501, 502, 503, 504, 505, 506};
    auto sig3 = generator.compute_signature(doc3);
    index.insert(1003, sig3);

    // Query candidates for doc1
    std::vector<uint64_t> candidates;
    index.query_candidates(sig1, candidates);

    // Candidate set must contain item 1001 itself (exact match in all bands)
    EXPECT_FALSE(candidates.empty());
    bool found_self = false;
    for (uint64_t id : candidates) {
        if (id == 1001)
            found_self = true;
    }
    EXPECT_TRUE(found_self);

    // Test clear()
    index.clear();
    candidates.clear();
    index.query_candidates(sig1, candidates);
    EXPECT_TRUE(candidates.empty());
}

TEST(LSHIndexTest, SimHashFingerprintIndexing) {
    // 4 bands, 16 rows each = 64 bits
    LSHIndex<uint64_t, 4, 16> index;

    uint64_t fp1 = 0x123456789ABCDEF0ULL;
    uint64_t fp2 = 0x123456789ABC0000ULL; // matches in bands 2 and 3
    uint64_t fp3 = 0xFEDCBA9876543210ULL; // completely different

    index.insert(1, fp1);
    index.insert(2, fp2);
    index.insert(3, fp3);

    auto candidates = index.query_candidates(fp1);
    EXPECT_FALSE(candidates.empty());

    bool found_1 = false;
    bool found_2 = false;
    for (uint64_t id : candidates) {
        if (id == 1)
            found_1 = true;
        if (id == 2)
            found_2 = true;
    }
    EXPECT_TRUE(found_1);
    EXPECT_TRUE(found_2);
}

TEST(SimHashTest, SimHash128BasicOperations) {
    std::vector<uint64_t> lo = {1, 2, 3, 4, 5};
    std::vector<uint64_t> hi = {10, 20, 30, 40, 50};

    SimHash128 fp1 = SimHash::compute_128(lo, hi);
    SimHash128 fp2 = SimHash::compute_128(lo, hi);

    EXPECT_EQ(fp1, fp2);
    EXPECT_EQ(SimHash::hamming_distance(fp1, fp2), 0U);
    EXPECT_NEAR(SimHash::cosine_similarity(fp1, fp2), 1.0, 1e-6);

    SimHash128 fp_diff{.low = ~fp1.low, .high = ~fp1.high};
    EXPECT_EQ(SimHash::hamming_distance(fp1, fp_diff), 128U);
    EXPECT_NEAR(SimHash::cosine_similarity(fp1, fp_diff), -1.0, 1e-6);
}

TEST(LSHIndexTest, MultiProbeRadius1HammingPerturbation) {
    // 4 bands, 16 bits per band
    LSHIndex<uint64_t, 4, 16> index;

    uint64_t base_fp = 0x1111222233334444ULL;
    // Perturbed by 1 bit in band 0 (bit 0 flipped)
    uint64_t perturbed_fp = base_fp ^ 0x1ULL;

    index.insert(42, base_fp);

    // Exact query on perturbed_fp misses in band 0, but might collide in other bands if identical
    // With a 1-bit perturbation across all bands:
    uint64_t all_perturbed = base_fp ^ 0x0001000100010001ULL; // flips bit 0 in all 4 bands
    index.insert(99, all_perturbed);

    std::vector<uint64_t> candidates_exact;
    index.query_candidates(base_fp, candidates_exact);
    // 99 should not match exact query because all 4 bands differ by 1 bit
    bool found_99_exact = false;
    for (uint64_t c : candidates_exact) {
        if (c == 99)
            found_99_exact = true;
    }
    EXPECT_FALSE(found_99_exact);

    // With multi-probe radius 1, band 0 perturbs bit 0 and recovers item 99!
    std::vector<uint64_t> candidates_multiprobe;
    index.query_candidates_multiprobe(base_fp, candidates_multiprobe, 1);
    bool found_99_multiprobe = false;
    for (uint64_t c : candidates_multiprobe) {
        if (c == 99)
            found_99_multiprobe = true;
    }
    EXPECT_TRUE(found_99_multiprobe);
}

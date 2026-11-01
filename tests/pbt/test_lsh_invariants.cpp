#include "algoat/sketches/lsh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

using namespace algoat::sketches;

// ============================================================================
// 1. BBitMinHash: Metric and Estimation Invariants
// ============================================================================

RC_GTEST_PROP(LSHInvariantsPBT, MinHashSelfSimilarity, (const std::vector<uint64_t>& tokens)) {
    RC_PRE(!tokens.empty());

    BBitMinHash<128, 2> minhash(100);
    auto sig = minhash.compute_signature(tokens);

    RC_ASSERT(sig.hamming_distance(sig) == 0U);
    RC_ASSERT(sig.jaccard_similarity(sig) == 1.0);
}

RC_GTEST_PROP(LSHInvariantsPBT, MinHashSymmetryAndBounds,
              (const std::vector<uint64_t>& tokens_a, const std::vector<uint64_t>& tokens_b)) {
    BBitMinHash<128, 2> minhash(200);
    auto sig_a = minhash.compute_signature(tokens_a);
    auto sig_b = minhash.compute_signature(tokens_b);

    // Symmetry
    RC_ASSERT(sig_a.hamming_distance(sig_b) == sig_b.hamming_distance(sig_a));
    RC_ASSERT(sig_a.jaccard_similarity(sig_b) == sig_b.jaccard_similarity(sig_a));

    // Bounds
    double sim = sig_a.jaccard_similarity(sig_b);
    RC_ASSERT(sim >= 0.0);
    RC_ASSERT(sim <= 1.0);
}

RC_GTEST_PROP(LSHInvariantsPBT, MinHashTriangleInequality,
              (const std::vector<uint64_t>& tokens_a, const std::vector<uint64_t>& tokens_b,
               const std::vector<uint64_t>& tokens_c)) {
    BBitMinHash<64, 2> minhash(300);
    auto sig_a = minhash.compute_signature(tokens_a);
    auto sig_b = minhash.compute_signature(tokens_b);
    auto sig_c = minhash.compute_signature(tokens_c);

    size_t d_ab = sig_a.hamming_distance(sig_b);
    size_t d_bc = sig_b.hamming_distance(sig_c);
    size_t d_ac = sig_a.hamming_distance(sig_c);

    RC_ASSERT(d_ac <= d_ab + d_bc);
}

// ============================================================================
// 2. SimHash: Metric and Cosine Invariants
// ============================================================================

RC_GTEST_PROP(LSHInvariantsPBT, SimHashSelfSimilarityAndSymmetry,
              (const std::vector<uint64_t>& tokens_a, const std::vector<uint64_t>& tokens_b)) {
    uint64_t fp_a = SimHash::compute_64(tokens_a);
    uint64_t fp_b = SimHash::compute_64(tokens_b);

    // Self similarity
    RC_ASSERT(SimHash::hamming_distance(fp_a, fp_a) == 0U);
    RC_ASSERT(std::abs(SimHash::cosine_similarity(fp_a, fp_a) - 1.0) < 1e-6);

    // Symmetry
    RC_ASSERT(SimHash::hamming_distance(fp_a, fp_b) == SimHash::hamming_distance(fp_b, fp_a));
    RC_ASSERT(std::abs(SimHash::cosine_similarity(fp_a, fp_b) -
                       SimHash::cosine_similarity(fp_b, fp_a)) < 1e-6);

    // Cosine Bounds
    double cos_sim = SimHash::cosine_similarity(fp_a, fp_b);
    RC_ASSERT(cos_sim >= -1.0 - 1e-6);
    RC_ASSERT(cos_sim <= 1.0 + 1e-6);
}

RC_GTEST_PROP(LSHInvariantsPBT, SimHashHammingTriangleInequality,
              (uint64_t fp_a, uint64_t fp_b, uint64_t fp_c)) {
    uint32_t d_ab = SimHash::hamming_distance(fp_a, fp_b);
    uint32_t d_bc = SimHash::hamming_distance(fp_b, fp_c);
    uint32_t d_ac = SimHash::hamming_distance(fp_a, fp_c);

    RC_ASSERT(d_ac <= d_ab + d_bc);
}

// ============================================================================
// 3. LSHIndex: Candidate Retrieval and Deduplication Invariants
// ============================================================================

RC_GTEST_PROP(LSHInvariantsPBT, LSHIndexRecallAndUniqueness,
              (const std::vector<std::vector<uint64_t>>& doc_batches)) {
    RC_PRE(!doc_batches.empty());

    using MinHash = BBitMinHash<64, 2>;
    MinHash generator(400);
    LSHIndex<MinHash::Signature, 4, 16> index;

    std::vector<std::pair<uint64_t, MinHash::Signature>> indexed_items;
    uint64_t current_id = 1;

    for (const auto& doc : doc_batches) {
        if (doc.empty())
            continue;
        auto sig = generator.compute_signature(doc);
        index.insert(current_id, sig);
        indexed_items.emplace_back(current_id, sig);
        ++current_id;
    }

    // Verify each indexed item finds itself among candidate matches
    for (const auto& [id, sig] : indexed_items) {
        std::vector<uint64_t> candidates;
        index.query_candidates(sig, candidates);

        RC_ASSERT(!candidates.empty());
        // Verify output is sorted
        RC_ASSERT(std::is_sorted(candidates.begin(), candidates.end()));

        // Verify output has no duplicates
        auto it = std::adjacent_find(candidates.begin(), candidates.end());
        RC_ASSERT(it == candidates.end());

        // Verify query item itself is present
        RC_ASSERT(std::find(candidates.begin(), candidates.end(), id) != candidates.end());
    }
}

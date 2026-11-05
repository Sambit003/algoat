#include "algoat/dp/levenshtein.hpp"

#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using algoat::dp::Alignment;
using algoat::dp::EditOp;
using algoat::dp::lcs_length;
using algoat::dp::levenshtein_align;
using algoat::dp::levenshtein_batch;
using algoat::dp::levenshtein_distance;
using algoat::dp::levenshtein_within_k;

// Helper to verify that operations accurately transform pattern into text
std::string apply_alignment(std::string_view pattern, const std::vector<EditOp>& ops) {
    std::string result;
    size_t p_idx = 0;
    for (EditOp op : ops) {
        switch (op) {
        case EditOp::Match:
            result.push_back(pattern[p_idx++]);
            break;
        case EditOp::Substitute:
            // Pattern char is consumed, substituted char comes from alignment
            // We note that in alignment, the resulting string should match text.
            // We verify by consuming pattern[p_idx++].
            p_idx++;
            break;
        case EditOp::Insert:
            // Character inserted into pattern (not advancing pattern index)
            break;
        case EditOp::Delete:
            p_idx++;
            break;
        }
    }
    return result;
}

// Full verification of Alignment operations against pattern and text
void verify_alignment_transforms(std::string_view pattern, std::string_view text,
                                 const Alignment& align) {
    size_t i = 0;
    size_t j = 0;
    size_t cost = 0;
    std::string reconstructed;

    for (EditOp op : align.operations) {
        switch (op) {
        case EditOp::Match:
            ASSERT_LT(i, pattern.size());
            ASSERT_LT(j, text.size());
            EXPECT_EQ(pattern[i], text[j]);
            reconstructed.push_back(pattern[i]);
            ++i;
            ++j;
            break;
        case EditOp::Substitute:
            ASSERT_LT(i, pattern.size());
            ASSERT_LT(j, text.size());
            EXPECT_NE(pattern[i], text[j]);
            reconstructed.push_back(text[j]);
            ++i;
            ++j;
            ++cost;
            break;
        case EditOp::Insert:
            ASSERT_LT(j, text.size());
            reconstructed.push_back(text[j]);
            ++j;
            ++cost;
            break;
        case EditOp::Delete:
            ASSERT_LT(i, pattern.size());
            ++i;
            ++cost;
            break;
        }
    }

    EXPECT_EQ(i, pattern.size());
    EXPECT_EQ(j, text.size());
    EXPECT_EQ(reconstructed, text);
    EXPECT_EQ(cost, align.distance);
}

// ============================================================================
// 1. levenshtein_distance Tests
// ============================================================================

TEST(LevenshteinTest, EmptyStrings) {
    EXPECT_EQ(levenshtein_distance("", ""), 0);
    EXPECT_EQ(levenshtein_distance("", "hello"), 5);
    EXPECT_EQ(levenshtein_distance("world", ""), 5);
}

TEST(LevenshteinTest, IdenticalStrings) {
    EXPECT_EQ(levenshtein_distance("a", "a"), 0);
    EXPECT_EQ(levenshtein_distance("algoat", "algoat"), 0);
    std::string long_str(100, 'x');
    EXPECT_EQ(levenshtein_distance(long_str, long_str), 0);
}

TEST(LevenshteinTest, ClassicalExamples) {
    EXPECT_EQ(levenshtein_distance("kitten", "sitting"), 3);
    EXPECT_EQ(levenshtein_distance("flaw", "lawn"), 2);
    EXPECT_EQ(levenshtein_distance("saturday", "sunday"), 3);
    EXPECT_EQ(levenshtein_distance("gumbo", "gambol"), 2);
    EXPECT_EQ(levenshtein_distance("rosettacode", "raisethysword"), 8);
}

TEST(LevenshteinTest, SingleEditOperations) {
    EXPECT_EQ(levenshtein_distance("cat", "cot"), 1);  // Substitute
    EXPECT_EQ(levenshtein_distance("cat", "cats"), 1); // Insert
    EXPECT_EQ(levenshtein_distance("cats", "cat"), 1); // Delete
}

TEST(LevenshteinTest, ExactBoundaryLength64) {
    std::string p64(64, 'a');
    std::string t64(64, 'a');
    EXPECT_EQ(levenshtein_distance(p64, t64), 0);

    t64[63] = 'b';
    EXPECT_EQ(levenshtein_distance(p64, t64), 1);

    t64[0] = 'z';
    EXPECT_EQ(levenshtein_distance(p64, t64), 2);
}

TEST(LevenshteinTest, PatternsGreaterThan64) {
    // Tests 128-bit, 256-bit, and multi-block regimes
    std::string p100(100, 'a');
    std::string t100(100, 'a');
    t100[50] = 'b';
    t100[99] = 'c';
    EXPECT_EQ(levenshtein_distance(p100, t100), 2);

    std::string p250(250, 'x');
    std::string t250(250, 'x');
    t250[10] = 'y';
    t250[100] = 'z';
    t250[200] = 'w';
    EXPECT_EQ(levenshtein_distance(p250, t250), 3);

    std::string p500(500, 'm');
    std::string t500 = p500 + "suffix";
    EXPECT_EQ(levenshtein_distance(p500, t500), 6);
}

TEST(LevenshteinTest, AcceptanceCriteria_100kRandomPairs) {
    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> char_dist('a', 'd');
    std::uniform_int_distribution<size_t> len_dist(0, 32);

    for (size_t trial = 0; trial < 100000; ++trial) {
        const size_t m = len_dist(rng);
        const size_t n = len_dist(rng);

        std::string p;
        p.reserve(m);
        for (size_t i = 0; i < m; ++i)
            p.push_back(static_cast<char>(char_dist(rng)));

        std::string t;
        t.reserve(n);
        for (size_t i = 0; i < n; ++i)
            t.push_back(static_cast<char>(char_dist(rng)));

        // Naive DP
        size_t expected = 0;
        {
            std::vector<size_t> prev(m + 1);
            std::vector<size_t> curr(m + 1);
            for (size_t i = 0; i <= m; ++i)
                prev[i] = i;
            for (size_t j = 1; j <= n; ++j) {
                curr[0] = j;
                for (size_t i = 1; i <= m; ++i) {
                    size_t cost = (p[i - 1] == t[j - 1]) ? 0 : 1;
                    curr[i] = std::min({curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost});
                }
                std::swap(prev, curr);
            }
            expected = prev[m];
        }

        size_t actual = levenshtein_distance(p, t);
        ASSERT_EQ(actual, expected)
            << "Mismatch at trial " << trial << " for p='" << p << "', t='" << t << "'";
    }
}

// ============================================================================
// 2. levenshtein_within_k Tests
// ============================================================================

TEST(LevenshteinWithinKTest, LengthFilter) {
    EXPECT_EQ(levenshtein_within_k("abc", "abcdefgh", 2), 3);
    EXPECT_EQ(levenshtein_within_k("abcdefgh", "abc", 2), 3);
}

TEST(LevenshteinWithinKTest, KZero) {
    EXPECT_EQ(levenshtein_within_k("abc", "abc", 0), 0);
    EXPECT_EQ(levenshtein_within_k("abc", "abd", 0), 1);
    EXPECT_EQ(levenshtein_within_k("abc", "abcd", 0), 1);
}

TEST(LevenshteinWithinKTest, ThresholdBoundaries) {
    // "kitten" vs "sitting" has distance 3
    EXPECT_EQ(levenshtein_within_k("kitten", "sitting", 1), 2);
    EXPECT_EQ(levenshtein_within_k("kitten", "sitting", 2), 3);
    EXPECT_EQ(levenshtein_within_k("kitten", "sitting", 3), 3);
    EXPECT_EQ(levenshtein_within_k("kitten", "sitting", 5), 3);
}

TEST(LevenshteinWithinKTest, AffixTrimming) {
    std::string p = "COMMON_PREFIX_kitten_COMMON_SUFFIX";
    std::string t = "COMMON_PREFIX_sitting_COMMON_SUFFIX";
    EXPECT_EQ(levenshtein_within_k(p, t, 2), 3);
    EXPECT_EQ(levenshtein_within_k(p, t, 3), 3);
}

// ============================================================================
// 3. levenshtein_align Tests
// ============================================================================

TEST(LevenshteinAlignTest, EmptyCases) {
    auto a1 = levenshtein_align("", "");
    EXPECT_EQ(a1.distance, 0);
    EXPECT_TRUE(a1.operations.empty());

    auto a2 = levenshtein_align("", "xyz");
    EXPECT_EQ(a2.distance, 3);
    verify_alignment_transforms("", "xyz", a2);

    auto a3 = levenshtein_align("abc", "");
    EXPECT_EQ(a3.distance, 3);
    verify_alignment_transforms("abc", "", a3);
}

TEST(LevenshteinAlignTest, IdenticalStrings) {
    auto a = levenshtein_align("algoat", "algoat");
    EXPECT_EQ(a.distance, 0);
    verify_alignment_transforms("algoat", "algoat", a);
}

TEST(LevenshteinAlignTest, ClassicalAlignments) {
    auto a1 = levenshtein_align("kitten", "sitting");
    EXPECT_EQ(a1.distance, 3);
    verify_alignment_transforms("kitten", "sitting", a1);

    auto a2 = levenshtein_align("flaw", "lawn");
    EXPECT_EQ(a2.distance, 2);
    verify_alignment_transforms("flaw", "lawn", a2);

    auto a3 = levenshtein_align("saturday", "sunday");
    EXPECT_EQ(a3.distance, 3);
    verify_alignment_transforms("saturday", "sunday", a3);
}

TEST(LevenshteinAlignTest, LongerStrings) {
    std::string p = "The quick brown fox jumps over the lazy dog";
    std::string t = "That quick brown dog jumps over a lazy cat";
    auto a = levenshtein_align(p, t);
    EXPECT_EQ(a.distance, levenshtein_distance(p, t));
    verify_alignment_transforms(p, t, a);
}

// ============================================================================
// 4. lcs_length Tests
// ============================================================================

TEST(LcsLengthTest, EmptyAndTrivial) {
    EXPECT_EQ(lcs_length("", ""), 0);
    EXPECT_EQ(lcs_length("abc", ""), 0);
    EXPECT_EQ(lcs_length("", "xyz"), 0);
    EXPECT_EQ(lcs_length("abc", "abc"), 3);
    EXPECT_EQ(lcs_length("abc", "def"), 0);
}

TEST(LcsLengthTest, ClassicalExamples) {
    EXPECT_EQ(lcs_length("AGGTAB", "GXTXAYB"), 4);  // "GTAB"
    EXPECT_EQ(lcs_length("ABCBDAB", "BDCAB"), 4);   // "BCAB" or "BDAB"
    EXPECT_EQ(lcs_length("XMJYAUZ", "MZJAWXU"), 4); // "MJAU"
}

TEST(LcsLengthTest, LongerStrings) {
    std::string a(80, 'a');
    std::string b(90, 'a');
    EXPECT_EQ(lcs_length(a, b), 80);

    std::string a_mixed = "XYZ" + std::string(70, 'a') + "123";
    std::string b_mixed = "ABC" + std::string(70, 'a') + "456";
    EXPECT_EQ(lcs_length(a_mixed, b_mixed), 70);
}

// ============================================================================
// 5. levenshtein_batch Tests
// ============================================================================

TEST(LevenshteinBatchTest, EmptyBatch) {
    auto res = levenshtein_batch("pattern", {});
    EXPECT_TRUE(res.empty());
}

TEST(LevenshteinBatchTest, BatchOfTexts) {
    std::string pattern = "kitten";
    std::vector<std::string> targets = {
        "sitting", "kitten", "mitten", "kit", "kitchen", "k", "supercalifragilistic",
        "smitten", "bitten"};

    std::vector<std::string_view> target_views;
    for (const auto& s : targets)
        target_views.emplace_back(s);

    auto batch_res = levenshtein_batch(pattern, target_views);
    ASSERT_EQ(batch_res.size(), targets.size());

    for (size_t i = 0; i < targets.size(); ++i) {
        size_t expected = levenshtein_distance(pattern, targets[i]);
        EXPECT_EQ(batch_res[i], expected)
            << "Mismatch at index " << i << " for target " << targets[i];
    }
}

TEST(LevenshteinBatchTest, VaryingLengthsAVX2) {
    std::string pattern = "algoat";
    std::vector<std::string_view> targets = {
        "algo", "algoat_long_string_here", "", "a", "all_goat", "algoat", "goat", "algorithm"};

    auto batch_res = levenshtein_batch(pattern, targets);
    ASSERT_EQ(batch_res.size(), targets.size());

    for (size_t i = 0; i < targets.size(); ++i) {
        EXPECT_EQ(batch_res[i], levenshtein_distance(pattern, targets[i]));
    }
}

} // namespace

#include <algoat/dp/alignment.hpp>
#include <gtest/gtest.h>

using namespace algoat::dp;

TEST(AlignmentTest, ScoringMatrixBlosum62) {
    auto m = ScoringMatrix::blosum62();
    // Diagonal matches
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('A')][static_cast<uint8_t>('A')], 4);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('W')][static_cast<uint8_t>('W')], 11);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('C')][static_cast<uint8_t>('C')], 9);

    // Mismatches
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('A')][static_cast<uint8_t>('R')], -1);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('R')][static_cast<uint8_t>('A')], -1);

    // Case insensitivity
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('a')][static_cast<uint8_t>('a')], 4);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('a')][static_cast<uint8_t>('W')], -3);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('A')][static_cast<uint8_t>('w')], -3);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('w')][static_cast<uint8_t>('w')], 11);
}

TEST(AlignmentTest, ScoringMatrixPam250) {
    auto m = ScoringMatrix::pam250();
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('A')][static_cast<uint8_t>('A')], 2);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('W')][static_cast<uint8_t>('W')], 17);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('C')][static_cast<uint8_t>('C')], 12);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('a')][static_cast<uint8_t>('A')], 2);
}

TEST(AlignmentTest, ScoringMatrixDna) {
    auto m = ScoringMatrix::dna(2, -3);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('A')][static_cast<uint8_t>('A')], 2);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('a')][static_cast<uint8_t>('A')], 2);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('A')][static_cast<uint8_t>('C')], -3);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('G')][static_cast<uint8_t>('T')], -3);
    EXPECT_EQ(m.matrix[static_cast<uint8_t>('u')][static_cast<uint8_t>('U')], 2);
}

TEST(AlignmentTest, GapPenaltyDefaults) {
    GapPenalty gap{};
    EXPECT_EQ(gap.open, -5);
    EXPECT_EQ(gap.extend, -1);
}

TEST(AlignmentTest, NeedlemanWunschEmpty) {
    auto m = ScoringMatrix::blosum62();
    GapPenalty gap{-5, -1};

    auto res1 = needleman_wunsch("", "", m, gap);
    EXPECT_EQ(res1.score, 0);
    EXPECT_EQ(res1.aligned_query, "");
    EXPECT_EQ(res1.aligned_ref, "");
    EXPECT_EQ(res1.cigar, "");

    auto res2 = needleman_wunsch("HEAG", "", m, gap);
    // 4 characters inserted relative to ref: -(5 + 3*1) = -8
    EXPECT_EQ(res2.score, -8);
    EXPECT_EQ(res2.cigar, "4I");
    EXPECT_EQ(res2.aligned_query, "HEAG");
    EXPECT_EQ(res2.aligned_ref, "----");

    auto res3 = needleman_wunsch("", "PAW", m, gap);
    // 3 characters deleted from ref: -(5 + 2*1) = -7
    EXPECT_EQ(res3.score, -7);
    EXPECT_EQ(res3.cigar, "3D");
    EXPECT_EQ(res3.aligned_query, "---");
    EXPECT_EQ(res3.aligned_ref, "PAW");
}

TEST(AlignmentTest, NeedlemanWunschIdentical) {
    auto m = ScoringMatrix::blosum62();
    std::string s = "HEAGAWGHEE";
    auto res = needleman_wunsch(s, s, m, GapPenalty{-10, -1});
    EXPECT_GT(res.score, 0);
    EXPECT_EQ(res.cigar, "10M");
    EXPECT_EQ(res.aligned_query, s);
    EXPECT_EQ(res.aligned_ref, s);
    EXPECT_EQ(res.query_start, 0);
    EXPECT_EQ(res.query_end, 10);
    EXPECT_EQ(res.ref_start, 0);
    EXPECT_EQ(res.ref_end, 10);
}

TEST(AlignmentTest, NeedlemanWunschDnaAlignment) {
    auto m = ScoringMatrix::dna(2, -3);
    GapPenalty gap{-5, -2};
    // Match A with A, gap in ref for C, match G with G
    std::string q = "ACG";
    std::string r = "AG";
    auto res = needleman_wunsch(q, r, m, gap);
    // Best alignment:
    // Q: A C G
    // R: A - G
    // Score = 2 (match A-A) - 5 (gap open C) + 2 (match G-G) = -1
    EXPECT_EQ(res.score, -1);
    EXPECT_EQ(res.cigar, "1M1I1M");
    EXPECT_EQ(res.aligned_query, "ACG");
    EXPECT_EQ(res.aligned_ref, "A-G");
}

TEST(AlignmentTest, SmithWatermanEmpty) {
    auto m = ScoringMatrix::blosum62();
    GapPenalty gap{-5, -1};

    auto res = smith_waterman("", "PAW", m, gap);
    EXPECT_EQ(res.score, 0);
    EXPECT_EQ(res.cigar, "");

    EXPECT_EQ(smith_waterman_score("", "PAW", m, gap), 0);
    EXPECT_EQ(smith_waterman_score("PAW", "", m, gap), 0);
}

TEST(AlignmentTest, SmithWatermanExactSubstring) {
    auto m = ScoringMatrix::blosum62();
    GapPenalty gap{-10, -2};

    // Substring "WHEAE" is embedded inside flanking noise with negative scores (D vs W is -4)
    std::string query = "DDWHEAEDD";
    std::string ref = "WWWWHEAEWWWW";

    auto res = smith_waterman(query, ref, m, gap);
    EXPECT_GT(res.score, 0);
    EXPECT_EQ(res.cigar, "5M");
    EXPECT_EQ(res.aligned_query, "WHEAE");
    EXPECT_EQ(res.aligned_ref, "WHEAE");
    EXPECT_EQ(res.query_start, 2);
    EXPECT_EQ(res.query_end, 7);
    EXPECT_EQ(res.ref_start, 3);
    EXPECT_EQ(res.ref_end, 8);
}

TEST(AlignmentTest, SmithWatermanScoreSimdAgreement) {
    auto m = ScoringMatrix::blosum62();
    GapPenalty gap{-5, -1};

    // Sequences longer than 16 to trigger AVX2 SIMD path
    std::string q = "HEAGAWGHEEWWWWHEAGAWGHEE";
    std::string r = "PAWHEAEWHEAGAWGHEEWWWWHEAGAWGHEEPAW";

    auto local_res = smith_waterman(q, r, m, gap);
    int64_t score_simd = smith_waterman_score(q, r, m, gap);

    EXPECT_EQ(local_res.score, score_simd);
}

TEST(AlignmentTest, SmithWatermanBatch) {
    auto m = ScoringMatrix::blosum62();
    GapPenalty gap{-5, -1};

    std::string q = "HEAGAWGHEEWWWWHEAGAWGHEE";
    std::vector<std::string> targets = {"PAWHEAEWHEAGAWGHEEWWWWHEAGAWGHEEPAW",
                                        "NOTHINGINCOMMONATALL", "HEAGAWGHEEWWWWHEAGAWGHEE", ""};
    std::vector<std::string_view> target_views;
    for (const auto& t : targets) {
        target_views.emplace_back(t);
    }

    auto batch_scores = smith_waterman_batch(q, target_views, m, gap);
    ASSERT_EQ(batch_scores.size(), targets.size());

    for (size_t i = 0; i < targets.size(); ++i) {
        int64_t single_score = smith_waterman_score(q, targets[i], m, gap);
        EXPECT_EQ(batch_scores[i], single_score);
    }
}

TEST(AlignmentTest, ScoringMatrixCustomMatrix) {
    std::array<std::array<int8_t, 128>, 128> custom{};
    for (size_t i = 0; i < 128; ++i) {
        for (size_t j = 0; j < 128; ++j) {
            custom[i][j] = (i == j) ? 5 : -2;
        }
    }
    custom['A']['C'] = -7;
    custom['C']['A'] = -7;

    ScoringMatrix sm(5, -2, custom);
    EXPECT_EQ(sm.score('A', 'A'), 5);
    EXPECT_EQ(sm.score('A', 'C'), -7);
    EXPECT_EQ(sm.score('C', 'A'), -7);
    EXPECT_EQ(sm.score('A', 'G'), -2);
}

TEST(AlignmentTest, HirschbergLinearSpaceLargeSequence) {
    // 10^6 characters sequence aligned against 20 characters
    // Space is bounded by O(min(M, N)), consuming < 100 KB RAM
    std::string large_query(1'000'000, 'A');
    std::string short_ref(20, 'A');

    auto m = ScoringMatrix::dna(2, -3);
    GapPenalty gap{-5, -2};

    auto res = needleman_wunsch(large_query, short_ref, m, gap);
    EXPECT_EQ(res.query_end, 1'000'000);
    EXPECT_EQ(res.ref_end, 20);
    // 20 matches + (1'000'000 - 20) insertions into ref
    // Score: 20 * 2 - (5 + (1'000'000 - 21) * 2)
    int64_t expected_score = 20 * 2 - (5 + static_cast<int64_t>(1'000'000 - 21) * 2);
    EXPECT_EQ(res.score, expected_score);
}

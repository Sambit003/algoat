#include <algoat/dp/alignment.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

using algoat::dp::AlignmentResult;
using algoat::dp::GapPenalty;
using algoat::dp::needleman_wunsch;
using algoat::dp::ScoringMatrix;
using algoat::dp::smith_waterman;
using algoat::dp::smith_waterman_batch;
using algoat::dp::smith_waterman_score;

// Helper to reconstruct strings from CIGAR and verify integrity
bool verify_cigar_reconstruction(std::string_view query, std::string_view ref,
                                 const AlignmentResult& res) {
    if (res.cigar.empty()) {
        return query.empty() || ref.empty();
    }

    std::string q_recon;
    std::string r_recon;
    size_t num = 0;

    size_t qi = res.query_start;
    size_t ri = res.ref_start;

    for (char c : res.cigar) {
        if (c >= '0' && c <= '9') {
            num = num * 10 + static_cast<size_t>(c - '0');
        } else {
            for (size_t k = 0; k < num; ++k) {
                if (c == 'M') {
                    if (qi >= query.size() || ri >= ref.size())
                        return false;
                    q_recon.push_back(query[qi++]);
                    r_recon.push_back(ref[ri++]);
                } else if (c == 'I') {
                    if (qi >= query.size())
                        return false;
                    q_recon.push_back(query[qi++]);
                    r_recon.push_back('-');
                } else if (c == 'D') {
                    if (ri >= ref.size())
                        return false;
                    q_recon.push_back('-');
                    r_recon.push_back(ref[ri++]);
                }
            }
            num = 0;
        }
    }

    if (qi != res.query_end || ri != res.ref_end) {
        return false;
    }
    return (q_recon == res.aligned_query && r_recon == res.aligned_ref);
}

TEST(AlignmentPBT, GlobalAlignmentSymmetry) {
    rc::check("Needleman-Wunsch score is symmetric for symmetric scoring matrices", []() {
        const auto len_a = *rc::gen::inRange<size_t>(0, 30);
        const auto len_b = *rc::gen::inRange<size_t>(0, 30);

        std::string a, b;
        a.reserve(len_a);
        b.reserve(len_b);
        constexpr std::string_view alphabet = "ACGT";
        for (size_t i = 0; i < len_a; ++i) {
            a.push_back(alphabet[*rc::gen::inRange<size_t>(0, alphabet.size())]);
        }
        for (size_t i = 0; i < len_b; ++i) {
            b.push_back(alphabet[*rc::gen::inRange<size_t>(0, alphabet.size())]);
        }

        const auto matrix = ScoringMatrix::dna(2, -3);
        const GapPenalty gap{-5, -2};

        const auto res_ab = needleman_wunsch(a, b, matrix, gap);
        const auto res_ba = needleman_wunsch(b, a, matrix, gap);

        RC_ASSERT(res_ab.score == res_ba.score);
    });
}

TEST(AlignmentPBT, CigarReconstructionIntegrity) {
    rc::check("Aligned sequences and CIGAR reconstruct valid prefixes", []() {
        const auto len_a = *rc::gen::inRange<size_t>(1, 35);
        const auto len_b = *rc::gen::inRange<size_t>(1, 35);

        std::string a, b;
        constexpr std::string_view alphabet = "ACDEFGHIKLMNPQRSTVWY";
        for (size_t i = 0; i < len_a; ++i) {
            a.push_back(alphabet[*rc::gen::inRange<size_t>(0, alphabet.size())]);
        }
        for (size_t i = 0; i < len_b; ++i) {
            b.push_back(alphabet[*rc::gen::inRange<size_t>(0, alphabet.size())]);
        }

        const auto matrix = ScoringMatrix::blosum62();
        const GapPenalty gap{-6, -1};

        const auto nw_res = needleman_wunsch(a, b, matrix, gap);
        RC_ASSERT(verify_cigar_reconstruction(a, b, nw_res));

        const auto sw_res = smith_waterman(a, b, matrix, gap);
        if (sw_res.score > 0) {
            RC_ASSERT(verify_cigar_reconstruction(a, b, sw_res));
        }
    });
}

TEST(AlignmentPBT, SimdVsScalarEquivalence) {
    rc::check("Farrar SIMD local alignment score matches scalar DP", []() {
        // Generate sequences with lengths >= 16 to exercise AVX2 stripes
        const auto len_a = *rc::gen::inRange<size_t>(16, 60);
        const auto len_b = *rc::gen::inRange<size_t>(16, 60);

        std::string a, b;
        constexpr std::string_view alphabet = "ACDEFGHIKLMNPQRSTVWY";
        for (size_t i = 0; i < len_a; ++i) {
            a.push_back(alphabet[*rc::gen::inRange<size_t>(0, alphabet.size())]);
        }
        for (size_t i = 0; i < len_b; ++i) {
            b.push_back(alphabet[*rc::gen::inRange<size_t>(0, alphabet.size())]);
        }

        const auto matrix = ScoringMatrix::blosum62();
        const GapPenalty gap{-5, -1};

        const auto sw_full = smith_waterman(a, b, matrix, gap);
        const int64_t simd_score = smith_waterman_score(a, b, matrix, gap);

        RC_ASSERT(sw_full.score == simd_score);
    });
}

} // namespace

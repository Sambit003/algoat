#include <algoat/dp/alignment.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace {

using algoat::dp::AlignmentResult;
using algoat::dp::GapPenalty;
using algoat::dp::needleman_wunsch;
using algoat::dp::ScoringMatrix;
using algoat::dp::smith_waterman;
using algoat::dp::smith_waterman_score;

bool verify_cigar(std::string_view query, std::string_view ref, const AlignmentResult& res) {
    if (res.cigar.empty()) {
        return query.empty() || ref.empty();
    }

    size_t qi = res.query_start;
    size_t ri = res.ref_start;
    size_t num = 0;

    for (char c : res.cigar) {
        if (c >= '0' && c <= '9') {
            num = num * 10 + static_cast<size_t>(c - '0');
        } else {
            for (size_t k = 0; k < num; ++k) {
                if (c == 'M') {
                    if (qi >= query.size() || ri >= ref.size())
                        return false;
                    ++qi;
                    ++ri;
                } else if (c == 'I') {
                    if (qi >= query.size())
                        return false;
                    ++qi;
                } else if (c == 'D') {
                    if (ri >= ref.size())
                        return false;
                    ++ri;
                } else {
                    return false;
                }
            }
            num = 0;
        }
    }
    return (qi == res.query_end && ri == res.ref_end);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;
    }

    const size_t split = static_cast<size_t>(data[0]) % (size - 1);
    const size_t q_len = split;
    const size_t r_len = size - 1 - split;

    // Cap lengths to keep fuzzing throughput high
    if (q_len > 80 || r_len > 80) {
        return 0;
    }

    constexpr std::string_view alphabet = "ACDEFGHIKLMNPQRSTVWY";

    std::string query;
    std::string ref;
    query.reserve(q_len);
    ref.reserve(r_len);

    for (size_t i = 0; i < q_len; ++i) {
        query.push_back(alphabet[data[1 + i] % alphabet.size()]);
    }
    for (size_t i = 0; i < r_len; ++i) {
        ref.push_back(alphabet[data[1 + q_len + i] % alphabet.size()]);
    }

    const auto matrix = ScoringMatrix::blosum62();
    const GapPenalty gap{-5, -1};

    // Test Smith-Waterman scalar vs SIMD
    const auto sw_res = smith_waterman(query, ref, matrix, gap);
    const int64_t simd_score = smith_waterman_score(query, ref, matrix, gap);

    if (sw_res.score != simd_score) {
        __builtin_trap();
    }

    if (sw_res.score > 0 && !verify_cigar(query, ref, sw_res)) {
        __builtin_trap();
    }

    // Test Needleman-Wunsch Hirschberg
    const auto nw_res = needleman_wunsch(query, ref, matrix, gap);
    if (!verify_cigar(query, ref, nw_res)) {
        __builtin_trap();
    }

    return 0;
}

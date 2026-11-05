#include "algoat/dp/levenshtein.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

using algoat::dp::Alignment;
using algoat::dp::EditOp;
using algoat::dp::lcs_length;
using algoat::dp::levenshtein_align;
using algoat::dp::levenshtein_distance;
using algoat::dp::levenshtein_within_k;

size_t naive_levenshtein(std::string_view s1, std::string_view s2) {
    const size_t m = s1.size();
    const size_t n = s2.size();
    std::vector<std::vector<size_t>> dp(m + 1, std::vector<size_t>(n + 1));
    for (size_t i = 0; i <= m; ++i)
        dp[i][0] = i;
    for (size_t j = 0; j <= n; ++j)
        dp[0][j] = j;

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            const size_t cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost});
        }
    }
    return dp[m][n];
}

size_t naive_lcs(std::string_view s1, std::string_view s2) {
    const size_t m = s1.size();
    const size_t n = s2.size();
    std::vector<std::vector<size_t>> dp(m + 1, std::vector<size_t>(n + 1, 0));
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}

bool verify_alignment(std::string_view pattern, std::string_view text, const Alignment& align) {
    size_t i = 0;
    size_t j = 0;
    size_t cost = 0;
    std::string reconstructed;

    for (EditOp op : align.operations) {
        switch (op) {
        case EditOp::Match:
            if (i >= pattern.size() || j >= text.size() || pattern[i] != text[j])
                return false;
            reconstructed.push_back(pattern[i]);
            ++i;
            ++j;
            break;
        case EditOp::Substitute:
            if (i >= pattern.size() || j >= text.size())
                return false;
            reconstructed.push_back(text[j]);
            ++i;
            ++j;
            ++cost;
            break;
        case EditOp::Insert:
            if (j >= text.size())
                return false;
            reconstructed.push_back(text[j]);
            ++j;
            ++cost;
            break;
        case EditOp::Delete:
            if (i >= pattern.size())
                return false;
            ++i;
            ++cost;
            break;
        }
    }

    return (i == pattern.size()) && (j == text.size()) && (reconstructed == text) &&
           (cost == align.distance);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2)
        return 0;

    const size_t split = static_cast<size_t>(data[0]) % (size - 1);
    const size_t p_len = split;
    const size_t t_len = size - 1 - split;

    // Cap lengths to keep fuzzing throughput high against O(MN) naive check
    if (p_len > 120 || t_len > 120)
        return 0;

    const char* p_ptr = reinterpret_cast<const char*>(data + 1);
    const char* t_ptr = reinterpret_cast<const char*>(data + 1 + p_len);

    std::string_view pattern(p_ptr, p_len);
    std::string_view text(t_ptr, t_len);

    const size_t expected_dist = naive_levenshtein(pattern, text);
    const size_t actual_dist = levenshtein_distance(pattern, text);
    if (actual_dist != expected_dist) {
        __builtin_trap();
    }

    const size_t k = static_cast<size_t>(data[0]) % 16;
    const size_t expected_within_k = std::min(expected_dist, k + 1);
    const size_t actual_within_k = levenshtein_within_k(pattern, text, k);
    if (actual_within_k != expected_within_k) {
        __builtin_trap();
    }

    const auto align = levenshtein_align(pattern, text);
    if (align.distance != actual_dist || !verify_alignment(pattern, text, align)) {
        __builtin_trap();
    }

    const size_t expected_lcs = naive_lcs(pattern, text);
    const size_t actual_lcs = lcs_length(pattern, text);
    if (actual_lcs != expected_lcs) {
        __builtin_trap();
    }

    return 0;
}

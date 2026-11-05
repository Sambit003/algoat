#include "algoat/dp/levenshtein.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
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

// Reference Wagner-Fischer O(MN) DP
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

// Reference naive LCS length DP
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

// Metric space properties
TEST(LevenshteinPBT, MetricSpaceInvariants) {
    rc::check("Levenshtein distance satisfies metric space properties",
              [](const std::string& a, const std::string& b, const std::string& c) {
                  // Non-negativity & Identity
                  RC_ASSERT(levenshtein_distance(a, a) == 0);
                  if (a != b) {
                      RC_ASSERT(levenshtein_distance(a, b) > 0);
                  }

                  // Symmetry
                  RC_ASSERT(levenshtein_distance(a, b) == levenshtein_distance(b, a));

                  // Triangle inequality: d(a, c) <= d(a, b) + d(b, c)
                  RC_ASSERT(levenshtein_distance(a, c) <=
                            levenshtein_distance(a, b) + levenshtein_distance(b, c));
              });
}

// Differential testing against naive DP
TEST(LevenshteinPBT, DifferentialTestingAgainstNaiveDP) {
    rc::check("levenshtein_distance matches reference DP", []() {
        const auto len_a = *rc::gen::inRange<size_t>(0, 80);
        const auto len_b = *rc::gen::inRange<size_t>(0, 80);

        std::string a, b;
        a.reserve(len_a);
        b.reserve(len_b);
        for (size_t i = 0; i < len_a; ++i) {
            a.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 4))));
        }
        for (size_t i = 0; i < len_b; ++i) {
            b.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 4))));
        }

        const size_t expected = naive_levenshtein(a, b);
        const size_t actual = levenshtein_distance(a, b);
        RC_ASSERT(actual == expected);
    });
}

// Ukkonen bounded distance equivalence
TEST(LevenshteinPBT, BoundedWithinKEquivalence) {
    rc::check("levenshtein_within_k matches min(distance, k + 1)",
              [](const std::string& a, const std::string& b) {
                  const size_t full_dist = levenshtein_distance(a, b);
                  const size_t k = *rc::gen::inRange<size_t>(0, 15);

                  const size_t expected = std::min(full_dist, k + 1);
                  const size_t actual = levenshtein_within_k(a, b, k);
                  RC_ASSERT(actual == expected);
              });
}

// Alignment reconstruction invariant
TEST(LevenshteinPBT, AlignmentReconstruction) {
    rc::check("levenshtein_align operations transform pattern to text with exact cost", []() {
        const auto len_p = *rc::gen::inRange<size_t>(0, 70);
        const auto len_t = *rc::gen::inRange<size_t>(0, 70);

        std::string p, t;
        for (size_t i = 0; i < len_p; ++i) {
            p.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 5))));
        }
        for (size_t i = 0; i < len_t; ++i) {
            t.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 5))));
        }

        const auto align = levenshtein_align(p, t);
        RC_ASSERT(align.distance == levenshtein_distance(p, t));
        RC_ASSERT(verify_alignment(p, t, align));
    });
}

// LCS length property-based testing
TEST(LevenshteinPBT, LcsDifferentialTesting) {
    rc::check("lcs_length matches naive LCS DP and symmetry", []() {
        const auto len_a = *rc::gen::inRange<size_t>(0, 80);
        const auto len_b = *rc::gen::inRange<size_t>(0, 80);

        std::string a, b;
        for (size_t i = 0; i < len_a; ++i) {
            a.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 4))));
        }
        for (size_t i = 0; i < len_b; ++i) {
            b.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 4))));
        }

        const size_t expected = naive_lcs(a, b);
        const size_t actual = lcs_length(a, b);
        RC_ASSERT(actual == expected);
        RC_ASSERT(lcs_length(b, a) == expected);
        RC_ASSERT(2 * actual <= a.size() + b.size());
    });
}

// Batch edit distance testing
TEST(LevenshteinPBT, BatchEquivalence) {
    rc::check("levenshtein_batch matches individual levenshtein_distance",
              [](const std::string& pattern, const std::vector<std::string>& targets) {
                  std::vector<std::string_view> views;
                  views.reserve(targets.size());
                  for (const auto& s : targets)
                      views.emplace_back(s);

                  const auto batch_res = levenshtein_batch(pattern, views);
                  RC_ASSERT(batch_res.size() == targets.size());
                  for (size_t i = 0; i < targets.size(); ++i) {
                      RC_ASSERT(batch_res[i] == levenshtein_distance(pattern, targets[i]));
                  }
              });
}

} // namespace

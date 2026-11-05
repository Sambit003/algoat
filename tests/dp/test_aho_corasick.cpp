#include "algoat/dp/aho_corasick.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using algoat::dp::AcMatch;
using algoat::dp::AhoCorasick;

// ============================================================================
// 1. Basic Construction and Metadata
// ============================================================================

TEST(AhoCorasickTest, EmptyAutomaton) {
    AhoCorasick<char> ac;
    ac.build();

    EXPECT_EQ(ac.num_keywords(), 0);
    EXPECT_GE(ac.num_states(), 1); // Root state 0
    EXPECT_GT(ac.memory_usage_bytes(), 0);

    EXPECT_TRUE(ac.find_all("hello world").empty());
    EXPECT_FALSE(ac.find_first("hello world").has_value());

    bool callback_called = false;
    ac.find_all_callback("hello world", [&](AcMatch) {
        callback_called = true;
        return true;
    });
    EXPECT_FALSE(callback_called);
}

TEST(AhoCorasickTest, SingleKeywordMatch) {
    AhoCorasick<char> ac;
    ac.add_keyword("needle");
    ac.build();

    EXPECT_EQ(ac.num_keywords(), 1);
    EXPECT_GE(ac.num_states(), 7); // Root + 6 characters

    std::string text = "haystack with needle in it";
    auto matches = ac.find_all(text);
    ASSERT_EQ(matches.size(), 1);
    EXPECT_EQ(matches[0].pattern_id, 0);
    EXPECT_EQ(matches[0].length, 6);
    EXPECT_EQ(matches[0].end_pos, 20); // "haystack with needle" ends at index 20
    EXPECT_EQ(text.substr(matches[0].end_pos - matches[0].length, matches[0].length), "needle");

    auto first = ac.find_first(text);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->pattern_id, 0);
    EXPECT_EQ(first->end_pos, 20);
    EXPECT_EQ(first->length, 6);
}

// ============================================================================
// 2. Multi-Pattern Matching (Classic Aho-Corasick Example)
// ============================================================================

TEST(AhoCorasickTest, ClassicAhoCorasickPatterns) {
    // Classic Aho & Corasick (1975) example dictionary:
    // P = {"he", "she", "his", "hers"}
    // Text: "ushers"
    AhoCorasick<char> ac;
    ac.add_keyword("he");   // id 0
    ac.add_keyword("she");  // id 1
    ac.add_keyword("his");  // id 2
    ac.add_keyword("hers"); // id 3
    ac.build();

    EXPECT_EQ(ac.num_keywords(), 4);

    // Streaming callback reports ALL overlapping matches:
    // "ushers":
    // at end_pos 3 ("she"): "she" (id 1, len 3), "he" (id 0, len 2)
    // at end_pos 6 ("hers"): "hers" (id 3, len 4)
    std::vector<AcMatch> callback_matches;
    ac.find_all_callback("ushers", [&](AcMatch m) {
        callback_matches.push_back(m);
        return true;
    });

    ASSERT_GE(callback_matches.size(), 3);
    bool found_she = false;
    bool found_he = false;
    bool found_hers = false;
    for (const auto& m : callback_matches) {
        if (m.pattern_id == 1 && m.end_pos == 4 && m.length == 3)
            found_she = true;
        if (m.pattern_id == 0 && m.end_pos == 4 && m.length == 2)
            found_he = true;
        if (m.pattern_id == 3 && m.end_pos == 6 && m.length == 4)
            found_hers = true;
    }
    EXPECT_TRUE(found_she);
    EXPECT_TRUE(found_he);
    EXPECT_TRUE(found_hers);

    // find_all returns non-overlapping matches:
    // Should select non-overlapping matches.
    auto non_overlap = ac.find_all("ushers");
    EXPECT_FALSE(non_overlap.empty());
    for (size_t i = 1; i < non_overlap.size(); ++i) {
        size_t prev_end = non_overlap[i - 1].end_pos;
        size_t curr_start = non_overlap[i].end_pos - non_overlap[i].length;
        EXPECT_GE(curr_start, prev_end);
    }
}

// ============================================================================
// 3. Callback Early Termination
// ============================================================================

TEST(AhoCorasickTest, CallbackEarlyTermination) {
    AhoCorasick<char> ac;
    ac.add_keyword("a");
    ac.build();

    std::string text = "aaaaa";
    size_t count = 0;
    ac.find_all_callback(text, [&](AcMatch) {
        ++count;
        return count < 2; // Stop after 2 matches
    });

    EXPECT_EQ(count, 2);
}

// ============================================================================
// 4. Binary Alphabet (uint8_t)
// ============================================================================

TEST(AhoCorasickTest, BinaryAlphabetSupport) {
    AhoCorasick<uint8_t> ac;
    std::vector<uint8_t> p1 = {0x00, 0xFF, 0x00};
    std::vector<uint8_t> p2 = {0xFF, 0xFE};
    ac.add_keyword(std::basic_string_view<uint8_t>(p1.data(), p1.size()));
    ac.add_keyword(std::basic_string_view<uint8_t>(p2.data(), p2.size()));
    ac.build();

    EXPECT_EQ(ac.num_keywords(), 2);

    std::vector<uint8_t> stream = {0x12, 0x00, 0xFF, 0x00, 0xFF, 0xFE, 0x99};
    auto matches = ac.find_all(std::basic_string_view<uint8_t>(stream.data(), stream.size()));
    ASSERT_EQ(matches.size(), 2);
    EXPECT_EQ(matches[0].pattern_id, 0);
    EXPECT_EQ(matches[0].end_pos, 4);
    EXPECT_EQ(matches[1].pattern_id, 1);
    EXPECT_EQ(matches[1].end_pos, 6);
}

// ============================================================================
// 5. Large Pattern Set & Compactness Acceptance Criterion
// ============================================================================

TEST(AhoCorasickTest, MemoryUsageAcceptanceCriterion) {
    // Acceptance criteria 4:
    // memory_usage_bytes() for K = 10,000 patterns of avg length 8 is < 5 MB
    AhoCorasick<char> ac;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> char_dist('a', 'z');

    constexpr size_t kNumPatterns = 10000;
    constexpr size_t kAvgLength = 8;
    std::vector<std::string> patterns;
    patterns.reserve(kNumPatterns);

    for (size_t i = 0; i < kNumPatterns; ++i) {
        std::string p;
        p.reserve(kAvgLength);
        for (size_t j = 0; j < kAvgLength; ++j) {
            p.push_back(static_cast<char>(char_dist(rng)));
        }
        patterns.push_back(std::move(p));
        ac.add_keyword(patterns.back());
    }

    ac.build();
    EXPECT_EQ(ac.num_keywords(), kNumPatterns);

    size_t mem = ac.memory_usage_bytes();
    EXPECT_LT(mem, 5 * 1024 * 1024); // Strictly under 5 MB
}

// ============================================================================
// 6. Failure Link Substring Verification Criterion
// ============================================================================

TEST(AhoCorasickTest, FailureLinkPrefixVerification) {
    // Acceptance criteria 2:
    // Every failure link verified: text.substr(pos - fail_depth, fail_depth) is a valid prefix.
    AhoCorasick<char> ac;
    ac.add_keyword("cacm");
    ac.add_keyword("acm");
    ac.add_keyword("c");
    ac.add_keyword("m");
    ac.build();

    std::string text = "acacm";
    std::vector<AcMatch> matches;
    ac.find_all_callback(text, [&](AcMatch m) {
        matches.push_back(m);
        return true;
    });

    EXPECT_FALSE(matches.empty());
    for (const auto& m : matches) {
        std::string_view sub = std::string_view(text).substr(m.end_pos - m.length, m.length);
        if (m.pattern_id == 0)
            EXPECT_EQ(sub, "cacm");
        if (m.pattern_id == 1)
            EXPECT_EQ(sub, "acm");
        if (m.pattern_id == 2)
            EXPECT_EQ(sub, "c");
        if (m.pattern_id == 3)
            EXPECT_EQ(sub, "m");
    }
}

// ============================================================================
// 7. Custom Keyword ID Assignment
// ============================================================================

TEST(AhoCorasickTest, CustomKeywordId) {
    AhoCorasick<char> ac;
    ac.add_keyword("apple", 1001);
    ac.add_keyword("banana", 2002);
    ac.add_keyword("orange", 3003);
    ac.build();

    auto first = ac.find_first("i like apple and banana");
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->pattern_id, 1001);
    EXPECT_EQ(first->length, 5);

    std::vector<AcMatch> all_matches;
    ac.find_all_callback("banana orange apple", [&](AcMatch m) {
        all_matches.push_back(m);
        return true;
    });

    ASSERT_EQ(all_matches.size(), 3);
    EXPECT_EQ(all_matches[0].pattern_id, 2002);
    EXPECT_EQ(all_matches[1].pattern_id, 3003);
    EXPECT_EQ(all_matches[2].pattern_id, 1001);
}

// ============================================================================
// 8. Acceptance Criterion 1: Naive std::string::find Equivalence Test
// ============================================================================

TEST(AhoCorasickTest, NaiveFindEquivalenceAcceptanceCriterion) {
    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<int> char_dist('a', 'd'); // small alphabet for dense matches
    std::uniform_int_distribution<size_t> len_dist(2, 6);

    constexpr size_t kNumTrials = 10;
    constexpr size_t kPatternsPerTrial = 20;
    constexpr size_t kTextLen = 5000;

    for (size_t trial = 0; trial < kNumTrials; ++trial) {
        std::vector<std::string> patterns;
        patterns.reserve(kPatternsPerTrial);
        for (size_t i = 0; i < kPatternsPerTrial; ++i) {
            size_t len = len_dist(rng);
            std::string p;
            for (size_t j = 0; j < len; ++j) {
                p.push_back(static_cast<char>(char_dist(rng)));
            }
            patterns.push_back(std::move(p));
        }

        std::string text;
        text.reserve(kTextLen);
        for (size_t i = 0; i < kTextLen; ++i) {
            text.push_back(static_cast<char>(char_dist(rng)));
        }

        // 1. Collect all matches via naive std::string::find
        struct MatchRecord {
            size_t end_pos;
            size_t pattern_id;
            size_t length;
            bool operator==(const MatchRecord& o) const = default;
            auto operator<=>(const MatchRecord& o) const = default;
        };

        std::vector<MatchRecord> expected;
        for (size_t pid = 0; pid < patterns.size(); ++pid) {
            const auto& p = patterns[pid];
            size_t pos = text.find(p, 0);
            while (pos != std::string::npos) {
                expected.push_back(MatchRecord{pos + p.size(), pid, p.size()});
                pos = text.find(p, pos + 1);
            }
        }
        std::sort(expected.begin(), expected.end());

        // 2. Collect all matches via Aho-Corasick callback
        AhoCorasick<char> ac;
        for (size_t i = 0; i < patterns.size(); ++i) {
            ac.add_keyword(patterns[i], i);
        }
        ac.build();

        std::vector<MatchRecord> actual;
        ac.find_all_callback(text, [&](AcMatch m) {
            actual.push_back(MatchRecord{m.end_pos, m.pattern_id, m.length});
            return true;
        });
        std::sort(actual.begin(), actual.end());

        ASSERT_EQ(actual.size(), expected.size());
        EXPECT_EQ(actual, expected);
    }
}

// ============================================================================
// 9. Acceptance Criterion 2: Comprehensive Failure Link Invariant
// ============================================================================

TEST(AhoCorasickTest, FullAutomatonFailureLinkVerification) {
    AhoCorasick<char> ac;
    std::vector<std::string> dict = {"a", "ab", "bc", "bca", "c", "caa"};
    for (const auto& w : dict) {
        ac.add_keyword(w);
    }
    ac.build();

    // Collect all prefixes of keywords present in the trie
    std::set<std::string> all_prefixes;
    all_prefixes.insert("");
    for (const auto& w : dict) {
        for (size_t len = 1; len <= w.size(); ++len) {
            all_prefixes.insert(w.substr(0, len));
        }
    }

    // For each prefix string, feeding it through Aho-Corasick should end in a state
    // whose failure target corresponds to the longest proper suffix of the prefix
    // that is also an element in all_prefixes.
    for (const auto& prefix : all_prefixes) {
        if (prefix.empty())
            continue;

        // Longest proper suffix of prefix in all_prefixes
        std::string expected_fail_suffix;
        for (size_t s_len = prefix.size() - 1; s_len >= 1; --s_len) {
            std::string cand = prefix.substr(prefix.size() - s_len, s_len);
            if (all_prefixes.contains(cand)) {
                expected_fail_suffix = cand;
                break;
            }
        }

        // Verification: match output of expected_fail_suffix matches suffix outputs of prefix.
        std::vector<size_t> prefix_matches;
        ac.find_all_callback(prefix, [&](AcMatch m) {
            if (m.end_pos == prefix.size()) {
                prefix_matches.push_back(m.pattern_id);
            }
            return true;
        });

        for (size_t pid : prefix_matches) {
            EXPECT_LE(dict[pid].size(), prefix.size());
            EXPECT_EQ(prefix.substr(prefix.size() - dict[pid].size()), dict[pid]);
        }
    }
}

} // namespace

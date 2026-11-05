#include "algoat/dp/aho_corasick.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

using algoat::dp::AcMatch;
using algoat::dp::AhoCorasick;

// Reference: Naive finder for all occurrences of patterns
struct NaiveMatch {
    size_t start;
    size_t end;
    size_t pattern_id;
    size_t length;
};

std::vector<NaiveMatch> naive_find_all_occurrences(std::string_view text,
                                                   const std::vector<std::string>& keywords) {
    std::vector<NaiveMatch> matches;
    for (size_t pid = 0; pid < keywords.size(); ++pid) {
        const auto& kw = keywords[pid];
        if (kw.empty())
            continue;
        size_t pos = text.find(kw, 0);
        while (pos != std::string_view::npos) {
            matches.push_back({pos, pos + kw.size(), pid, kw.size()});
            pos = text.find(kw, pos + 1);
        }
    }
    std::sort(matches.begin(), matches.end(), [](const NaiveMatch& a, const NaiveMatch& b) {
        if (a.end != b.end)
            return a.end < b.end;
        return a.pattern_id < b.pattern_id;
    });
    return matches;
}

// Invariant 1: All callback matches correspond to exact pattern occurrences
TEST(AhoCorasickPBT, CallbackMatchesExactSubstrings) {
    rc::check("find_all_callback reports exact substring matches", []() {
        const auto num_kw = *rc::gen::inRange<size_t>(1, 10);
        std::vector<std::string> keywords;
        keywords.reserve(num_kw);

        AhoCorasick<char> ac;
        for (size_t i = 0; i < num_kw; ++i) {
            const auto kw_len = *rc::gen::inRange<size_t>(1, 8);
            std::string kw;
            for (size_t j = 0; j < kw_len; ++j) {
                kw.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 3))));
            }
            keywords.push_back(kw);
            ac.add_keyword(kw);
        }
        ac.build();

        const auto text_len = *rc::gen::inRange<size_t>(0, 100);
        std::string text;
        for (size_t i = 0; i < text_len; ++i) {
            text.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 3))));
        }

        std::vector<AcMatch> actual_matches;
        ac.find_all_callback(text, [&](AcMatch m) {
            actual_matches.push_back(m);
            return true;
        });

        // Verify every match is genuine
        for (const auto& m : actual_matches) {
            RC_ASSERT(m.pattern_id < keywords.size());
            RC_ASSERT(m.length == keywords[m.pattern_id].size());
            RC_ASSERT(m.end_pos <= text.size());
            RC_ASSERT(m.end_pos >= m.length);
            std::string_view sub(text.data() + m.end_pos - m.length, m.length);
            RC_ASSERT(sub == keywords[m.pattern_id]);
        }

        // Verify complete coverage: compare count against naive
        auto expected_matches = naive_find_all_occurrences(text, keywords);
        RC_ASSERT(actual_matches.size() == expected_matches.size());
    });
}

// Invariant 2: find_all produces strictly non-overlapping matches
TEST(AhoCorasickPBT, NonOverlappingMatchesInvariant) {
    rc::check("find_all produces valid, non-overlapping intervals", []() {
        const auto num_kw = *rc::gen::inRange<size_t>(1, 15);
        std::vector<std::string> keywords;
        keywords.reserve(num_kw);

        AhoCorasick<char> ac;
        for (size_t i = 0; i < num_kw; ++i) {
            const auto kw_len = *rc::gen::inRange<size_t>(1, 10);
            std::string kw;
            for (size_t j = 0; j < kw_len; ++j) {
                kw.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 4))));
            }
            keywords.push_back(kw);
            ac.add_keyword(kw);
        }
        ac.build();

        const auto text_len = *rc::gen::inRange<size_t>(0, 200);
        std::string text;
        for (size_t i = 0; i < text_len; ++i) {
            text.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 4))));
        }

        auto matches = ac.find_all(text);

        size_t last_end = 0;
        for (const auto& m : matches) {
            RC_ASSERT(m.pattern_id < keywords.size());
            RC_ASSERT(m.length == keywords[m.pattern_id].size());
            RC_ASSERT(m.end_pos <= text.size());
            RC_ASSERT(m.end_pos >= m.length);

            size_t start = m.end_pos - m.length;
            RC_ASSERT(start >= last_end); // Strictly non-overlapping
            last_end = m.end_pos;

            std::string_view sub(text.data() + start, m.length);
            RC_ASSERT(sub == keywords[m.pattern_id]);
        }

        // find_first consistency
        auto first = ac.find_first(text);
        if (matches.empty()) {
            RC_ASSERT(!first.has_value());
        } else {
            RC_ASSERT(first.has_value());
            RC_ASSERT(first->pattern_id == matches[0].pattern_id);
            RC_ASSERT(first->end_pos == matches[0].end_pos);
            RC_ASSERT(first->length == matches[0].length);
        }
    });
}

} // namespace

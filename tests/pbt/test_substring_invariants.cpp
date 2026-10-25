#include "algoat/strings/substring_search.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <string>
#include <string_view>
#include <vector>

TEST(SubstringPBT, MatchesStdStringFind) {
    rc::check("find_substring matches std::string_view::find",
              [](const std::string& haystack, const std::string& needle) {
                  std::string_view hs(haystack);
                  std::string_view ne(needle);

                  const auto expected = hs.find(ne);
                  const auto actual = algoat::strings::find_substring(hs, ne);
                  RC_ASSERT(actual == expected);

                  const auto scalar_actual = algoat::strings::detail::two_way_search(hs, ne);
                  RC_ASSERT(scalar_actual == expected);
              });
}

TEST(SubstringPBT, SmallAlphabetCollisions) {
    // Generate strings from small alphabet {'a', 'b', 'c'} to stress collisions and periodicity
    rc::check("Matches std::string_view::find on repetitive small alphabet strings", []() {
        const auto hs_len = *rc::gen::inRange<std::size_t>(0, 1000);
        const auto ne_len = *rc::gen::inRange<std::size_t>(0, 50);

        std::string haystack;
        haystack.reserve(hs_len);
        for (std::size_t i = 0; i < hs_len; ++i) {
            haystack.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 3))));
        }

        std::string needle;
        needle.reserve(ne_len);
        for (std::size_t i = 0; i < ne_len; ++i) {
            needle.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 3))));
        }

        std::string_view hs(haystack);
        std::string_view ne(needle);

        const auto expected = hs.find(ne);
        const auto actual = algoat::strings::find_substring(hs, ne);
        RC_ASSERT(actual == expected);

        const auto scalar_actual = algoat::strings::detail::two_way_search(hs, ne);
        RC_ASSERT(scalar_actual == expected);
    });
}

TEST(SubstringPBT, FindAllSubstringsInvariants) {
    rc::check("find_all_substrings finds all valid occurrences", []() {
        const auto hs_len = *rc::gen::inRange<std::size_t>(10, 500);
        const auto ne_len = *rc::gen::inRange<std::size_t>(1, 10);

        std::string haystack;
        haystack.reserve(hs_len);
        for (std::size_t i = 0; i < hs_len; ++i) {
            haystack.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 4))));
        }

        std::string needle;
        needle.reserve(ne_len);
        for (std::size_t i = 0; i < ne_len; ++i) {
            needle.push_back(static_cast<char>('a' + (*rc::gen::inRange<int>(0, 4))));
        }

        std::string_view hs(haystack);
        std::string_view ne(needle);

        const auto occurrences = algoat::strings::find_all_substrings(hs, ne);

        // Every reported match must indeed equal needle
        for (auto idx : occurrences) {
            RC_ASSERT(idx + ne.size() <= hs.size());
            RC_ASSERT(hs.substr(idx, ne.size()) == ne);
        }

        // Must be in strictly ascending order
        for (std::size_t i = 1; i < occurrences.size(); ++i) {
            RC_ASSERT(occurrences[i] > occurrences[i - 1]);
        }

        // Compare against naive brute-force collection
        std::vector<std::string_view::size_type> naive_expected;
        for (std::size_t i = 0; i + ne.size() <= hs.size(); ++i) {
            if (hs.substr(i, ne.size()) == ne) {
                naive_expected.push_back(i);
            }
        }
        RC_ASSERT(occurrences == naive_expected);
    });
}

TEST(SubstringPBT, SearchPatternRangeAdaptorMatches) {
    rc::check("search_pattern returns iterator matching std::string::find",
              [](const std::string& haystack, const std::string& needle) {
                  auto it = algoat::strings::search_pattern(haystack, needle);
                  const auto expected_pos = haystack.find(needle);

                  if (expected_pos == std::string::npos) {
                      RC_ASSERT(it == haystack.end());
                  } else {
                      RC_ASSERT(static_cast<std::size_t>(std::distance(haystack.begin(), it)) ==
                                expected_pos);
                  }
              });
}

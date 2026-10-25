#include "algoat/strings/substring_search.hpp"

#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

TEST(SubstringSearchTest, EmptyNeedle) {
    EXPECT_EQ(algoat::strings::find_substring("hello", ""), 0);
    EXPECT_EQ(algoat::strings::find_substring("", ""), 0);
}

TEST(SubstringSearchTest, EmptyHaystack) {
    EXPECT_EQ(algoat::strings::find_substring("", "a"), std::string_view::npos);
}

TEST(SubstringSearchTest, NeedleLongerThanHaystack) {
    EXPECT_EQ(algoat::strings::find_substring("hi", "hello"), std::string_view::npos);
}

TEST(SubstringSearchTest, SingleCharacter) {
    EXPECT_EQ(algoat::strings::find_substring("banana", "b"), 0);
    EXPECT_EQ(algoat::strings::find_substring("banana", "a"), 1);
    EXPECT_EQ(algoat::strings::find_substring("banana", "n"), 2);
    EXPECT_EQ(algoat::strings::find_substring("banana", "z"), std::string_view::npos);
}

TEST(SubstringSearchTest, TwoCharacter) {
    EXPECT_EQ(algoat::strings::find_substring("banana", "ba"), 0);
    EXPECT_EQ(algoat::strings::find_substring("banana", "an"), 1);
    EXPECT_EQ(algoat::strings::find_substring("banana", "na"), 2);
    EXPECT_EQ(algoat::strings::find_substring("banana", "no"), std::string_view::npos);
}

TEST(SubstringSearchTest, BasicPositions) {
    std::string_view text = "The quick brown fox jumps over the lazy dog";
    EXPECT_EQ(algoat::strings::find_substring(text, "The"), 0);
    EXPECT_EQ(algoat::strings::find_substring(text, "quick"), 4);
    EXPECT_EQ(algoat::strings::find_substring(text, "dog"), 40);
    EXPECT_EQ(algoat::strings::find_substring(text, "cat"), std::string_view::npos);
}

TEST(SubstringSearchTest, PeriodicNeedle) {
    std::string_view text = "abacababacab";
    EXPECT_EQ(algoat::strings::find_substring(text, "abacab"), 0);
    EXPECT_EQ(algoat::strings::find_substring(text, "acab"), 2);

    std::string_view repetitive = "aaaaab";
    EXPECT_EQ(algoat::strings::find_substring(repetitive, "aaab"), 2);
    EXPECT_EQ(algoat::strings::find_substring(repetitive, "aaa"), 0);
}

TEST(SubstringSearchTest, PathologicalRepetitive) {
    std::string haystack(10000, 'a');
    haystack += 'b';
    std::string needle(500, 'a');
    needle += 'b';

    EXPECT_EQ(algoat::strings::find_substring(haystack, needle), haystack.find(needle));
    EXPECT_EQ(algoat::strings::find_substring(haystack, needle), 9500);

    std::string needle_not_found(500, 'a');
    needle_not_found += 'c';
    EXPECT_EQ(algoat::strings::find_substring(haystack, needle_not_found), std::string_view::npos);
}

TEST(SubstringSearchTest, SimdChunkCrossing) {
    // Generate text spanning multiple SIMD chunks (> 128 bytes)
    std::string text;
    for (int i = 0; i < 50; ++i) {
        text += "0123456789abcdef";
    }
    // Place target near middle and near end
    text[340] = 'X';
    text[341] = 'Y';
    text[342] = 'Z';
    EXPECT_EQ(algoat::strings::find_substring(text, "XYZ"), 340);

    text[text.size() - 2] = 'Q';
    text[text.size() - 1] = 'W';
    EXPECT_EQ(algoat::strings::find_substring(text, "QW"), text.size() - 2);
}

TEST(SubstringSearchTest, FalsePositiveCollisionHandling) {
    // Pattern with same first and last characters but different middle
    std::string haystack = "axxxza---axxxza---aMATCHz---axxxza";
    std::string needle = "aMATCHz";
    EXPECT_EQ(algoat::strings::find_substring(haystack, needle), 18);
}

TEST(SubstringSearchTest, FindAllSubstrings) {
    std::string_view text = "banana";
    auto occurrences = algoat::strings::find_all_substrings(text, "an");
    std::vector<std::string_view::size_type> expected = {1, 3};
    EXPECT_EQ(occurrences, expected);

    auto overlapping = algoat::strings::find_all_substrings("aaaaa", "aa");
    std::vector<std::string_view::size_type> expected_overlapping = {0, 1, 2, 3};
    EXPECT_EQ(overlapping, expected_overlapping);

    EXPECT_TRUE(algoat::strings::find_all_substrings(text, "xyz").empty());
    EXPECT_TRUE(algoat::strings::find_all_substrings(text, "").empty());
}

TEST(SubstringSearchTest, SearchPatternRangeAdaptor) {
    std::string text = "Modern C++20 ranges pattern search";
    std::string needle = "ranges";

    auto it = algoat::strings::search_pattern(text, needle);
    ASSERT_NE(it, text.end());
    EXPECT_EQ(std::string_view(&*it, needle.size()), "ranges");

    std::string missing = "missing";
    auto it_missing = algoat::strings::search_pattern(text, missing);
    EXPECT_EQ(it_missing, text.end());

    std::vector<char> text_vec(text.begin(), text.end());
    std::vector<char> needle_vec(needle.begin(), needle.end());
    auto it_vec = algoat::strings::search_pattern(text_vec, needle_vec);
    ASSERT_NE(it_vec, text_vec.end());
    EXPECT_EQ(std::distance(text_vec.begin(), it_vec), text.find(needle));
    EXPECT_EQ(std::distance(text_vec.begin(), it_vec), 13);
}

TEST(SubstringSearchTest, RvalueNonBorrowedRangeReturnsDangling) {
    // Non-borrowed rvalue generic range returns std::ranges::dangling per C++20 ranges contract
    auto it =
        algoat::strings::search_pattern(std::vector<int>{1, 2, 3, 4, 5}, std::vector<int>{3, 4});
    static_assert(std::same_as<decltype(it), std::ranges::dangling>);
}

TEST(SubstringSearchTest, TwoWayScalarDirect) {
    // Directly verify the scalar Crochemore-Perrin Two-Way engine
    EXPECT_EQ(algoat::strings::detail::two_way_search("hello world", "world"), 6);
    EXPECT_EQ(algoat::strings::detail::two_way_search("banana", "an"), 1);
    EXPECT_EQ(algoat::strings::detail::two_way_search("banana", "na"), 2);
    EXPECT_EQ(algoat::strings::detail::two_way_search("banana", "z"), std::string_view::npos);
    EXPECT_EQ(algoat::strings::detail::two_way_search("", "a"), std::string_view::npos);
    EXPECT_EQ(algoat::strings::detail::two_way_search("a", ""), 0);

    // Periodicity verification
    std::size_t period = 0;
    std::size_t ell = algoat::strings::detail::critical_factorization("abacab", period);
    EXPECT_GT(period, 0);
    EXPECT_LE(ell, 6);
}

} // namespace

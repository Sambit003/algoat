#include <algoat/succinct/detail/sais.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using algoat::succinct::detail::build_suffix_array;
using algoat::succinct::detail::build_suffix_array_with_sentinel;

// Helper: Naive brute-force suffix array construction for ground truth verification
std::vector<size_t> naive_suffix_array(std::string_view s) {
    size_t n = s.size();
    std::vector<size_t> sa(n);
    std::iota(sa.begin(), sa.end(), 0);
    std::sort(sa.begin(), sa.end(), [s](size_t a, size_t b) { return s.substr(a) < s.substr(b); });
    return sa;
}

} // namespace

TEST(SaisTest, EmptyAndSingleChar) {
    // Empty text
    {
        std::vector<uint8_t> empty;
        auto sa = build_suffix_array(std::span<const uint8_t>(empty));
        EXPECT_TRUE(sa.empty());
    }

    // Single character
    {
        std::string text = "a";
        auto sa = build_suffix_array(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
        ASSERT_EQ(sa.size(), 1);
        EXPECT_EQ(sa[0], 0);
    }
}

TEST(SaisTest, KnownStrings) {
    const std::vector<std::string> test_strings = {
        "banana", "abracadabra", "mississippi", "mmiissiissiippii", "cacba",
        "ba",     "ab",          "aaaaa",       "abababab"};

    for (const auto& str : test_strings) {
        auto expected = naive_suffix_array(str);
        auto actual = build_suffix_array(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(str.data()), str.size()));

        ASSERT_EQ(actual.size(), expected.size()) << "Failed on string: " << str;
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(actual[i], expected[i])
                << "Mismatch at index " << i << " for string \"" << str << "\"";
        }
    }
}

TEST(SaisTest, RandomizedDnaSequences) {
    std::mt19937_64 rng(42);
    const char bases[] = {'A', 'C', 'G', 'T'};
    std::uniform_int_distribution<size_t> base_dist(0, 3);

    for (size_t len : {10, 50, 100, 500, 1000}) {
        std::string dna;
        dna.reserve(len);
        for (size_t i = 0; i < len; ++i) {
            dna.push_back(bases[base_dist(rng)]);
        }

        auto expected = naive_suffix_array(dna);
        auto actual = build_suffix_array(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(dna.data()), dna.size()));

        ASSERT_EQ(actual.size(), expected.size());
        EXPECT_EQ(actual, expected) << "Mismatch on DNA sequence of length " << len;
    }
}

#include "algoat/strings/burstsort.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

TEST(BurstsortTest, EmptyAndSingleElement) {
    std::vector<std::string> empty_vec;
    algoat::strings::burstsort(std::span<std::string>{empty_vec});
    EXPECT_TRUE(empty_vec.empty());

    std::vector<std::string> single_vec = {"hello"};
    algoat::strings::burstsort(std::span<std::string>{single_vec});
    EXPECT_EQ(single_vec.size(), 1);
    EXPECT_EQ(single_vec[0], "hello");

    std::vector<std::string_view> single_sv = {"world"};
    algoat::strings::burstsort(std::span<std::string_view>{single_sv});
    EXPECT_EQ(single_sv[0], "world");
}

TEST(BurstsortTest, SmallCorpus) {
    std::vector<std::string> words = {"banana", "apple",      "cherry",  "date",
                                      "fig",    "elderberry", "apricot", "blueberry"};
    auto expected = words;
    std::sort(expected.begin(), expected.end());

    algoat::strings::burstsort(std::span<std::string>{words});
    EXPECT_EQ(words, expected);
}

TEST(BurstsortTest, StringViewSpan) {
    std::vector<std::string_view> urls = {"https://algoat.dev/docs/intro",
                                          "https://algoat.dev/docs/sorting",
                                          "https://algoat.dev/api/v1/sort",
                                          "https://algoat.dev/api/v2/sort",
                                          "http://algoat.dev/",
                                          "ftp://mirror.algoat.dev/releases",
                                          "https://algoat.dev/about"};
    auto expected = urls;
    std::sort(expected.begin(), expected.end());

    algoat::strings::burstsort(std::span<std::string_view>{urls});
    EXPECT_EQ(urls, expected);
}

TEST(BurstsortTest, LongSharedPrefixes) {
    std::vector<std::string> paths = {"/usr/local/include/algoat/strings/burstsort.hpp",
                                      "/usr/local/include/algoat/strings/multikey_quicksort.hpp",
                                      "/usr/local/include/algoat/sorting/pdqsort.hpp",
                                      "/usr/local/include/algoat/algoat.hpp",
                                      "/usr/local/bin/algoat-bench",
                                      "/usr/bin/gcc",
                                      "/usr/bin/clang",
                                      "/etc/algoat.json"};
    auto expected = paths;
    std::sort(expected.begin(), expected.end());

    algoat::strings::burstsort(std::span<std::string>{paths});
    EXPECT_EQ(paths, expected);
}

TEST(BurstsortTest, HeavyDuplicatesPreventInfiniteBurst) {
    // 20,000 identical strings (exceeds L = 8,192) to verify robust burst termination
    std::vector<std::string> duplicates(20000, "same_common_prefix_string_value_12345");
    auto expected = duplicates;

    algoat::strings::burstsort(std::span<std::string>{duplicates});
    EXPECT_EQ(duplicates, expected);
}

TEST(BurstsortTest, LargeCorpusExceedingBurstThreshold) {
    // 25,000 strings to guarantee multiple bursting levels across L = 8,192
    std::mt19937_64 rng(1337);
    std::vector<std::string> data;
    data.reserve(25000);
    const std::string charset = "abcdefghijklmnopqrstuvwxyz";

    for (int i = 0; i < 25000; ++i) {
        int len = 5 + (rng() % 25);
        std::string s;
        s.reserve(len);
        for (int j = 0; j < len; ++j) {
            s.push_back(charset[rng() % charset.size()]);
        }
        data.push_back(std::move(s));
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::strings::burstsort(std::span<std::string>{data});
    EXPECT_EQ(data, expected);
}

TEST(BurstsortTest, GenomicDNASequences) {
    std::mt19937_64 rng(999);
    std::vector<std::string> dna;
    dna.reserve(10000);
    const char bases[] = {'A', 'C', 'G', 'T'};

    for (int i = 0; i < 10000; ++i) {
        std::string seq;
        seq.reserve(32);
        for (int j = 0; j < 32; ++j) {
            seq.push_back(bases[rng() % 4]);
        }
        dna.push_back(std::move(seq));
    }

    auto expected = dna;
    std::sort(expected.begin(), expected.end());

    algoat::strings::burstsort(std::span<std::string>{dna});
    EXPECT_EQ(dna, expected);
}

TEST(BurstsortTest, NonAsciiAndEmbeddedNull) {
    std::vector<std::string> data = {std::string("alpha\0beta", 10),
                                     std::string("alpha", 5),
                                     std::string("alpha\0gamma", 11),
                                     "café",
                                     "cafe",
                                     "résumé",
                                     "resume",
                                     "\xFF\xFE\xFD"};
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::strings::burstsort(std::span<std::string>{data});
    EXPECT_EQ(data, expected);
}

TEST(BurstsortTest, RangeOverload) {
    std::vector<std::string> words = {"zebra", "yak", "xenon", "walrus"};
    auto expected = words;
    std::sort(expected.begin(), expected.end());

    algoat::strings::burstsort(words);
    EXPECT_EQ(words, expected);
}

TEST(BurstsortTest, CArrayRange) {
    std::string arr[4] = {"zebra", "yak", "xenon", "walrus"};
    algoat::strings::burstsort(arr);
    EXPECT_EQ(arr[0], "walrus");
    EXPECT_EQ(arr[1], "xenon");
    EXPECT_EQ(arr[2], "yak");
    EXPECT_EQ(arr[3], "zebra");
}

TEST(BurstsortTest, DeepIdenticalPrefix) {
    // 10,000 strings sharing a 500-character prefix
    const std::string prefix(500, 'k');
    std::vector<std::string> data;
    data.reserve(10000);
    for (int i = 0; i < 10000; ++i) {
        data.push_back(prefix + std::to_string(9999 - i));
    }
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::strings::burstsort(std::span<std::string>{data});
    EXPECT_EQ(data, expected);
}

} // namespace

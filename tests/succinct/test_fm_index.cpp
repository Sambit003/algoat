#include <algoat/succinct/fm_index.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using algoat::succinct::FMIndex;
using algoat::succinct::FMIndexConfig;

// Helper: Naive brute force locate in string
std::vector<size_t> naive_locate(std::string_view text, std::string_view pattern) {
    std::vector<size_t> positions;
    if (pattern.empty() || text.empty()) {
        return positions;
    }
    size_t pos = text.find(pattern, 0);
    while (pos != std::string_view::npos) {
        positions.push_back(pos);
        pos = text.find(pattern, pos + 1);
    }
    return positions;
}

} // namespace

TEST(FMIndexTest, EmptyTextHandling) {
    FMIndex idx{""};
    EXPECT_EQ(idx.text_length(), 0);
    EXPECT_EQ(idx.count(""), 0);
    EXPECT_EQ(idx.count("a"), 0);
    EXPECT_TRUE(idx.locate("a").empty());
    EXPECT_EQ(idx.extract(0, 0), "");
    auto [lo, hi] = idx.backward_search("a");
    EXPECT_EQ(lo, hi);
}

TEST(FMIndexTest, SingleCharText) {
    FMIndex idx{"a"};
    EXPECT_EQ(idx.text_length(), 1);
    EXPECT_EQ(idx.count("a"), 1);
    EXPECT_EQ(idx.locate("a"), (std::vector<size_t>{0}));
    EXPECT_EQ(idx.count("b"), 0);
    EXPECT_TRUE(idx.locate("b").empty());
    EXPECT_EQ(idx.extract(0, 1), "a");
}

TEST(FMIndexTest, BananaFullTextSearch) {
    const std::string text = "banana";
    FMIndex idx{text};
    EXPECT_EQ(idx.text_length(), 6);

    // Single character queries
    EXPECT_EQ(idx.count("a"), 3);
    EXPECT_EQ(idx.locate("a"), (std::vector<size_t>{1, 3, 5}));

    EXPECT_EQ(idx.count("b"), 1);
    EXPECT_EQ(idx.locate("b"), (std::vector<size_t>{0}));

    EXPECT_EQ(idx.count("n"), 2);
    EXPECT_EQ(idx.locate("n"), (std::vector<size_t>{2, 4}));

    // Substring queries
    EXPECT_EQ(idx.count("ana"), 2);
    EXPECT_EQ(idx.locate("ana"), (std::vector<size_t>{1, 3}));

    EXPECT_EQ(idx.count("anana"), 1);
    EXPECT_EQ(idx.locate("anana"), (std::vector<size_t>{1}));

    EXPECT_EQ(idx.count("banana"), 1);
    EXPECT_EQ(idx.locate("banana"), (std::vector<size_t>{0}));

    EXPECT_EQ(idx.count("nan"), 1);
    EXPECT_EQ(idx.locate("nan"), (std::vector<size_t>{2}));

    // Non-occurring queries (including null byte that must NOT match sentinel)
    EXPECT_EQ(idx.count(std::string_view("\0", 1)), 0);
    EXPECT_TRUE(idx.locate(std::string_view("\0", 1)).empty());

    EXPECT_EQ(idx.count("bananas"), 0);
    EXPECT_TRUE(idx.locate("bananas").empty());

    EXPECT_EQ(idx.count("xyz"), 0);
    EXPECT_TRUE(idx.locate("xyz").empty());

    EXPECT_EQ(idx.count("c"), 0);
    EXPECT_TRUE(idx.locate("c").empty());

    // Backward search interval checks
    auto [lo, hi] = idx.backward_search("ana");
    EXPECT_EQ(hi - lo, 2);
}

TEST(FMIndexTest, ArbitraryByteSequencesWithNullBytes) {
    // Text containing embedded null bytes and binary values 0..255
    std::vector<uint8_t> raw_bytes = {0x00, 'h', 'e', 'l', 'l',  'o',  0x00, 'w',
                                      'o',  'r', 'l', 'd', 0x00, 0xFF, 0xFE, 0x00};
    FMIndex idx{std::span<const uint8_t>(raw_bytes.data(), raw_bytes.size())};
    EXPECT_EQ(idx.text_length(), raw_bytes.size());

    // Query for null byte
    const char null_pat[] = {'\0'};
    std::string_view null_view(null_pat, 1);
    EXPECT_EQ(idx.count(null_view), 4);
    EXPECT_EQ(idx.locate(null_view), (std::vector<size_t>{0, 6, 12, 15}));

    // Query for sub-patterns
    std::string_view hello("hello");
    EXPECT_EQ(idx.count(hello), 1);
    EXPECT_EQ(idx.locate(hello), (std::vector<size_t>{1}));

    // Query across embedded null byte
    const char cross_pat[] = {'o', '\0', 'w'};
    std::string_view cross_view(cross_pat, 3);
    EXPECT_EQ(idx.count(cross_view), 1);
    EXPECT_EQ(idx.locate(cross_view), (std::vector<size_t>{5}));

    // Query binary 0xFF, 0xFE
    const char bin_pat[] = {static_cast<char>(0xFF), static_cast<char>(0xFE)};
    std::string_view bin_view(bin_pat, 2);
    EXPECT_EQ(idx.count(bin_view), 1);
    EXPECT_EQ(idx.locate(bin_view), (std::vector<size_t>{13}));

    // Extraction across null bytes
    std::string extracted = idx.extract(5, 4);
    EXPECT_EQ(extracted.size(), 4);
    EXPECT_EQ(extracted[0], 'o');
    EXPECT_EQ(extracted[1], '\0');
    EXPECT_EQ(extracted[2], 'w');
    EXPECT_EQ(extracted[3], 'o');
}

TEST(FMIndexTest, ExtractSubstrings) {
    const std::string text = "abracadabra";
    FMIndex idx{text};

    EXPECT_EQ(idx.extract(0, text.size()), text);
    EXPECT_EQ(idx.extract(0, 4), "abra");
    EXPECT_EQ(idx.extract(4, 5), "cadab");
    EXPECT_EQ(idx.extract(7, 4), "abra");
    EXPECT_EQ(idx.extract(10, 1), "a");
    EXPECT_EQ(idx.extract(3, 0), "");
}

TEST(FMIndexTest, DifferentSampleRates) {
    const std::string text = "mississippi_and_missouri_river";
    for (size_t rate : {1, 2, 4, 8, 16, 32}) {
        FMIndexConfig cfg;
        cfg.sa_sample_rate = rate;
        cfg.build_isa_samples = true;
        FMIndex idx{text, cfg};

        for (std::string_view pat : {"issi", "ss", "river", "m", "z", "mississippi"}) {
            auto expected = naive_locate(text, pat);
            auto actual = idx.locate(pat);
            std::sort(actual.begin(), actual.end());
            EXPECT_EQ(actual, expected) << "Failed for rate=" << rate << " pat=" << pat;
            EXPECT_EQ(idx.count(pat), expected.size());
        }

        // Test extract
        EXPECT_EQ(idx.extract(0, text.size()), text);
        EXPECT_EQ(idx.extract(4, 7), text.substr(4, 7));
    }
}

TEST(FMIndexTest, RandomDnaGroundTruthStress) {
    std::mt19937_64 rng(12345);
    const char bases[] = {'A', 'C', 'G', 'T'};
    std::uniform_int_distribution<size_t> base_dist(0, 3);

    const size_t text_len = 10'000;
    std::string dna;
    dna.reserve(text_len);
    for (size_t i = 0; i < text_len; ++i) {
        dna.push_back(bases[base_dist(rng)]);
    }

    FMIndex idx{dna};
    EXPECT_EQ(idx.text_length(), text_len);

    std::uniform_int_distribution<size_t> pos_dist(0, text_len - 20);
    std::uniform_int_distribution<size_t> len_dist(2, 8);

    for (size_t q = 0; q < 500; ++q) {
        size_t p = pos_dist(rng);
        size_t l = len_dist(rng);
        std::string pattern = dna.substr(p, l);

        auto expected_locs = naive_locate(dna, pattern);
        EXPECT_EQ(idx.count(pattern), expected_locs.size());

        auto actual_locs = idx.locate(pattern);
        std::sort(actual_locs.begin(), actual_locs.end());
        EXPECT_EQ(actual_locs, expected_locs);

        // Verify each returned pos matches
        for (size_t pos : actual_locs) {
            EXPECT_EQ(dna.substr(pos, l), pattern);
        }

        // Test extraction
        EXPECT_EQ(idx.extract(p, l), pattern);
    }
}

TEST(FMIndexTest, LargeDnaGroundTruthStressOneMillionBases) {
    // Acceptance criterion: Verified against naive std::string::find on random patterns up to 10^6
    // bases
    std::mt19937_64 rng(54321);
    const char bases[] = {'A', 'C', 'G', 'T'};
    std::uniform_int_distribution<size_t> base_dist(0, 3);

    const size_t text_len = 1'000'000;
    std::string dna;
    dna.reserve(text_len);
    for (size_t i = 0; i < text_len; ++i) {
        dna.push_back(bases[base_dist(rng)]);
    }

    FMIndexConfig cfg;
    cfg.sa_sample_rate = 32;
    cfg.build_isa_samples = true;
    FMIndex idx{dna, cfg};
    EXPECT_EQ(idx.text_length(), text_len);

    // Verify memory efficiency: < 2 bytes per input character for s=32
    EXPECT_LT(idx.memory_usage_bytes(), 2 * text_len);

    std::uniform_int_distribution<size_t> pos_dist(0, text_len - 50);
    std::uniform_int_distribution<size_t> len_dist(6, 16);

    for (size_t q = 0; q < 100; ++q) {
        size_t p = pos_dist(rng);
        size_t l = len_dist(rng);
        std::string pattern = dna.substr(p, l);

        auto expected_locs = naive_locate(dna, pattern);
        EXPECT_EQ(idx.count(pattern), expected_locs.size());

        auto actual_locs = idx.locate(pattern);
        std::sort(actual_locs.begin(), actual_locs.end());
        EXPECT_EQ(actual_locs, expected_locs);

        // Verify substring extract
        EXPECT_EQ(idx.extract(p, l), pattern);
    }
}

TEST(FMIndexTest, SerializationFidelity) {
    const std::string text = "the_quick_brown_fox_jumps_over_the_lazy_dog_12345";
    FMIndex orig{text};

    std::vector<uint8_t> bytes = orig.serialize();
    EXPECT_FALSE(bytes.empty());

    FMIndex loaded = FMIndex::deserialize(bytes);
    EXPECT_EQ(loaded.text_length(), orig.text_length());
    EXPECT_EQ(loaded.count("the"), orig.count("the"));
    EXPECT_EQ(loaded.locate("the"), orig.locate("the"));
    EXPECT_EQ(loaded.extract(4, 15), orig.extract(4, 15));
    EXPECT_EQ(loaded.extract(0, text.size()), text);
}

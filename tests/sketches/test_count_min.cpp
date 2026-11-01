#include "algoat/sketches/count_min.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace algoat::sketches;

// Shared Zipf distribution helper for CDF generation and sampling
struct ZipfDistribution {
    std::vector<double> cdf;

    explicit ZipfDistribution(std::size_t num_keys, double alpha = 1.2) : cdf(num_keys) {
        double sum = 0.0;
        for (std::size_t i = 1; i <= num_keys; ++i) {
            sum += 1.0 / std::pow(static_cast<double>(i), alpha);
            cdf[i - 1] = sum;
        }
        for (std::size_t i = 0; i < num_keys; ++i) {
            cdf[i] /= sum;
        }
    }

    [[nodiscard]] uint64_t sample(uint64_t& rng_state) const {
        rng_state = detail::splitmix64(rng_state);
        double r =
            static_cast<double>(rng_state & 0xFFFFFFFFULL) / static_cast<double>(0x100000000ULL);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        return static_cast<uint64_t>(std::distance(cdf.begin(), it) + 1);
    }
};

// ============================================================================
// Slice 1: Construction, Parameter Validation, Sizing, Memory, Clear
// ============================================================================

TEST(CountMinSketchTest, DefaultConstructorParameters) {
    CountMinSketch<uint64_t, uint32_t> cms;
    // epsilon = 0.001 -> w = ceil(e / 0.001) = 2719
    // delta = 0.01 -> d = ceil(ln(100)) = 5
    EXPECT_EQ(cms.width(), 2719ULL);
    EXPECT_EQ(cms.depth(), 5ULL);
    EXPECT_EQ(cms.memory_bytes(), 2719ULL * 5ULL * sizeof(uint32_t));
}

TEST(CountMinSketchTest, CustomParametersDimensions) {
    // epsilon = 0.01 -> ceil(e / 0.01) = 272
    // delta = 0.05 -> ceil(ln(20)) = 3
    CountMinSketch<uint64_t, uint32_t> cms(0.01, 0.05, false);
    EXPECT_EQ(cms.width(), 272ULL);
    EXPECT_EQ(cms.depth(), 3ULL);
    EXPECT_EQ(cms.memory_bytes(), 272ULL * 3ULL * sizeof(uint32_t));
}

TEST(CountMinSketchTest, RejectsInvalidEpsilon) {
    EXPECT_THROW((CountMinSketch<uint64_t, uint32_t>(0.0, 0.01)), std::invalid_argument);
    EXPECT_THROW((CountMinSketch<uint64_t, uint32_t>(-0.1, 0.01)), std::invalid_argument);
    EXPECT_THROW((CountMinSketch<uint64_t, uint32_t>(1.0, 0.01)), std::invalid_argument);
    EXPECT_THROW((CountMinSketch<uint64_t, uint32_t>(1.5, 0.01)), std::invalid_argument);
    EXPECT_THROW(
        (CountMinSketch<uint64_t, uint32_t>(std::numeric_limits<double>::quiet_NaN(), 0.01)),
        std::invalid_argument);
    EXPECT_THROW(
        (CountMinSketch<uint64_t, uint32_t>(std::numeric_limits<double>::infinity(), 0.01)),
        std::invalid_argument);
    EXPECT_THROW((CountMinSketch<uint64_t, uint32_t>(1e-15, 0.01)), std::invalid_argument);
}

TEST(CountMinSketchTest, RejectsInvalidDelta) {
    EXPECT_THROW((CountMinSketch<uint64_t, uint32_t>(0.001, 0.0)), std::invalid_argument);
    EXPECT_THROW((CountMinSketch<uint64_t, uint32_t>(0.001, -0.05)), std::invalid_argument);
    EXPECT_THROW((CountMinSketch<uint64_t, uint32_t>(0.001, 1.0)), std::invalid_argument);
    EXPECT_THROW((CountMinSketch<uint64_t, uint32_t>(0.001, 2.0)), std::invalid_argument);
    EXPECT_THROW(
        (CountMinSketch<uint64_t, uint32_t>(0.001, std::numeric_limits<double>::quiet_NaN())),
        std::invalid_argument);
    EXPECT_THROW(
        (CountMinSketch<uint64_t, uint32_t>(0.001, std::numeric_limits<double>::infinity())),
        std::invalid_argument);
}

TEST(CountMinSketchTest, ClearResetsAllCounters) {
    CountMinSketch<uint64_t, uint32_t> cms(0.01, 0.05);
    cms.add(10, 100);
    EXPECT_GE(cms.estimate(10), 100);
    cms.clear();
    EXPECT_EQ(cms.width(), 272ULL);
    EXPECT_EQ(cms.depth(), 3ULL);
    EXPECT_EQ(cms.estimate(10), 0);
}

// ============================================================================
// Slice 2: Add, Estimate, Non-Underestimation, Error Bound, Conservative Update
// ============================================================================

TEST(CountMinSketchTest, BasicAddAndEstimate) {
    CountMinSketch<uint64_t, uint32_t> cms(0.001, 0.01, true);
    EXPECT_EQ(cms.estimate(42ULL), 0U);

    cms.add(42ULL, 1);
    EXPECT_EQ(cms.estimate(42ULL), 1U);

    cms.add(42ULL, 9);
    EXPECT_EQ(cms.estimate(42ULL), 10U);

    cms.add(99ULL, 5);
    EXPECT_EQ(cms.estimate(99ULL), 5U);
    EXPECT_EQ(cms.estimate(42ULL), 10U);
}

TEST(CountMinSketchTest, OneSidedNonUnderestimationInvariant) {
    CountMinSketch<uint64_t, uint32_t> cms(0.005, 0.01, true);
    std::unordered_map<uint64_t, uint32_t> exact;

    // Insert 1,000 distinct items with varied counts
    for (uint64_t i = 1; i <= 1000; ++i) {
        uint32_t count = static_cast<uint32_t>(i % 37 + 1);
        cms.add(i, count);
        exact[i] += count;
    }

    for (const auto& [key, true_f] : exact) {
        uint32_t est = cms.estimate(key);
        EXPECT_GE(est, true_f) << "Violation of non-underestimation invariant for key " << key;
    }
}

TEST(CountMinSketchTest, TheoreticalUpperBoundGuarantee) {
    // Width w = ceil(e / epsilon), epsilon = 0.01, delta = 0.05
    const double epsilon = 0.01;
    const double delta = 0.05;
    CountMinSketch<uint64_t, uint32_t> cms(epsilon, delta, false);

    std::unordered_map<uint64_t, uint32_t> exact;
    uint64_t total_weight = 0;

    // Generate stream of 20,000 items with non-uniform distribution
    for (uint64_t i = 0; i < 20000; ++i) {
        uint64_t key = (i * 2654435761ULL) % 500; // 500 distinct keys
        cms.add(key, 1);
        exact[key] += 1;
        total_weight += 1;
    }

    std::size_t violations = 0;
    const double max_allowed_error = epsilon * static_cast<double>(total_weight);

    for (const auto& [key, true_f] : exact) {
        uint32_t est = cms.estimate(key);
        if (static_cast<double>(est - true_f) > max_allowed_error) {
            violations++;
        }
    }

    double empirical_failure_rate =
        static_cast<double>(violations) / static_cast<double>(exact.size());
    EXPECT_LE(empirical_failure_rate, delta);
}

TEST(CountMinSketchTest, ConservativeUpdateBeatsStandardOnZipfSkew) {
    // Under Zipfian skew, Conservative Update strictly reduces mean overestimation error
    const double epsilon = 0.005;
    const double delta = 0.01;
    CountMinSketch<uint64_t, uint32_t> cms_cu(epsilon, delta, true);
    CountMinSketch<uint64_t, uint32_t> cms_std(epsilon, delta, false);

    const std::size_t num_keys = 1000;
    const std::size_t stream_len = 30000;
    ZipfDistribution zipf(num_keys, 1.2);

    std::unordered_map<uint64_t, uint32_t> exact;
    uint64_t rng = 0xdeadbeefULL;
    for (std::size_t i = 0; i < stream_len; ++i) {
        uint64_t key = zipf.sample(rng);
        cms_cu.add(key, 1);
        cms_std.add(key, 1);
        exact[key] += 1;
    }

    double total_err_cu = 0.0;
    double total_err_std = 0.0;

    for (const auto& [key, true_f] : exact) {
        uint32_t est_cu = cms_cu.estimate(key);
        uint32_t est_std = cms_std.estimate(key);

        EXPECT_GE(est_cu, true_f);
        EXPECT_GE(est_std, true_f);

        total_err_cu += static_cast<double>(est_cu - true_f);
        total_err_std += static_cast<double>(est_std - true_f);
    }

    double mean_err_cu = total_err_cu / static_cast<double>(exact.size());
    double mean_err_std = total_err_std / static_cast<double>(exact.size());

    // Conservative update must strictly beat standard CMS on Zipfian stream
    EXPECT_LT(mean_err_cu, mean_err_std);
}

// ============================================================================
// Slice 3: Merge, Dimension Check, Commutativity, Non-Underestimation, Saturation
// ============================================================================

TEST(CountMinSketchTest, MergeRejectsMismatchedDimensions) {
    CountMinSketch<uint64_t, uint32_t> cms1(0.01, 0.01);
    CountMinSketch<uint64_t, uint32_t> cms2(0.005, 0.01);
    CountMinSketch<uint64_t, uint32_t> cms3(0.01, 0.05);

    EXPECT_THROW(cms1.merge(cms2), std::invalid_argument);
    EXPECT_THROW(cms1.merge(cms3), std::invalid_argument);
}

TEST(CountMinSketchTest, MergeCommutativityStandardCMS) {
    CountMinSketch<uint64_t, uint32_t> a(0.01, 0.05, false);
    CountMinSketch<uint64_t, uint32_t> b(0.01, 0.05, false);

    for (uint64_t i = 1; i <= 200; ++i) {
        a.add(i, static_cast<uint32_t>(i));
        b.add(i + 100, static_cast<uint32_t>(i * 2));
    }

    CountMinSketch<uint64_t, uint32_t> a_merge_b = a;
    a_merge_b.merge(b);

    CountMinSketch<uint64_t, uint32_t> b_merge_a = b;
    b_merge_a.merge(a);

    for (uint64_t i = 1; i <= 350; ++i) {
        EXPECT_EQ(a_merge_b.estimate(i), b_merge_a.estimate(i));
    }
}

TEST(CountMinSketchTest, MergeNonUnderestimationPreservation) {
    CountMinSketch<uint64_t, uint32_t> a(0.01, 0.05, false);
    CountMinSketch<uint64_t, uint32_t> b(0.01, 0.05, false);

    a.add(42, 10);
    b.add(42, 25);
    a.add(100, 5);
    b.add(200, 7);

    CountMinSketch<uint64_t, uint32_t> merged = a;
    merged.merge(b);

    EXPECT_GE(merged.estimate(42), 35U);
    EXPECT_GE(merged.estimate(100), 5U);
    EXPECT_GE(merged.estimate(200), 7U);
}

TEST(CountMinSketchTest, MergeSaturatesWithoutOverflow) {
    CountMinSketch<uint64_t, uint32_t> a(0.01, 0.05, false);
    CountMinSketch<uint64_t, uint32_t> b(0.01, 0.05, false);

    uint32_t near_max = std::numeric_limits<uint32_t>::max() - 5;
    a.add(42, near_max);
    b.add(42, 100); // would overflow uint32_t

    a.merge(b);

    EXPECT_EQ(a.estimate(42), std::numeric_limits<uint32_t>::max());
}

// ============================================================================
// Slice 4: SpaceSavingHeavyHitters Stream-Summary, Top-k, Invariants, Zipf-100
// ============================================================================

TEST(SpaceSavingTest, ExactTrackingWhenDistinctAtMostK) {
    SpaceSavingHeavyHitters<uint64_t, 50> ss;
    EXPECT_TRUE(ss.top_k().empty());

    // Insert 30 distinct items (< 50)
    for (uint64_t i = 1; i <= 30; ++i) {
        ss.add(i, i * 10);
    }

    auto items = ss.top_k();
    EXPECT_EQ(items.size(), 30ULL);

    // Should be sorted descending by count
    for (std::size_t i = 0; i < items.size(); ++i) {
        EXPECT_EQ(items[i].error, 0ULL) << "Error must be 0 for distinct items <= K";
        uint64_t expected_key = 30 - i;
        EXPECT_EQ(items[i].key, expected_key);
        EXPECT_EQ(items[i].count, expected_key * 10);
    }
}

TEST(SpaceSavingTest, ConservationOfTotalStreamWeight) {
    SpaceSavingHeavyHitters<uint64_t, 20> ss;
    uint64_t total_weight = 0;

    // Stream with 100 distinct items (> K = 20)
    for (uint64_t i = 0; i < 5000; ++i) {
        uint64_t key = (i * 101ULL + 17ULL) % 100;
        uint64_t count = (i % 5) + 1;
        ss.add(key, count);
        total_weight += count;
    }

    auto items = ss.top_k();
    EXPECT_EQ(items.size(), 20ULL);

    uint64_t sum_counts = 0;
    for (const auto& item : items) {
        sum_counts += item.count;
    }

    EXPECT_EQ(sum_counts, total_weight) << "Sum of tracked counters must equal stream weight N";
}

TEST(SpaceSavingTest, ErrorBoundInvariant) {
    SpaceSavingHeavyHitters<uint64_t, 32> ss;
    std::unordered_map<uint64_t, uint64_t> exact;

    // Stream of 10,000 items over 200 distinct keys
    for (uint64_t i = 0; i < 10000; ++i) {
        uint64_t key = (i * 31337ULL) % 200;
        ss.add(key, 1);
        exact[key] += 1;
    }

    auto items = ss.top_k();
    for (const auto& item : items) {
        uint64_t true_f = exact[item.key];
        // Invariant: count - error <= true_f <= count
        EXPECT_GE(item.count, true_f) << "True frequency cannot exceed estimated count";
        EXPECT_LE(item.count - item.error, true_f) << "Under-bound violation for key " << item.key;
    }
}

TEST(SpaceSavingTest, Top100ZipfPrecisionAndRecall) {
    // Top-100 elements on Zipf stream (alpha = 1.2) recovered with 100% precision & recall
    // among 500 distinct keys where the top-100 have heavy volume and 400 tail keys exercise
    // eviction
    constexpr std::size_t K = 100;
    SpaceSavingHeavyHitters<uint64_t, K> ss;
    static_assert(std::is_same_v<SpaceSavingHeavyHitters<uint64_t, K>, SpaceSaving<uint64_t, K>>);

    ZipfDistribution zipf(K, 1.2);

    const std::size_t stream_len = 150000;
    std::unordered_map<uint64_t, uint64_t> exact;
    uint64_t rng = 0xc001cafeULL;

    // 400 tail keys (101..500) interleaved twice to trigger active evictions
    std::vector<uint64_t> tail_keys;
    tail_keys.reserve(800);
    for (int rep = 0; rep < 2; ++rep) {
        for (uint64_t k = 101; k <= 500; ++k) {
            tail_keys.push_back(k);
        }
    }
    std::size_t tail_idx = 0;
    const std::size_t tail_step = stream_len / tail_keys.size();

    for (std::size_t i = 0; i < stream_len; ++i) {
        uint64_t key = zipf.sample(rng);
        constexpr uint64_t zipf_weight = 10;
        ss.add(key, zipf_weight);
        exact[key] += zipf_weight;

        if (i % tail_step == 0 && tail_idx < tail_keys.size()) {
            uint64_t tk = tail_keys[tail_idx++];
            ss.add(tk, 1);
            exact[tk] += 1;
        }
    }

    // Determine the true top-100 keys
    std::vector<std::pair<uint64_t, uint64_t>> true_top(exact.begin(), exact.end());
    std::sort(true_top.begin(), true_top.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });
    true_top.resize(K);

    std::unordered_set<uint64_t> true_top_keys;
    for (const auto& [k, v] : true_top) {
        true_top_keys.insert(k);
    }

    auto ss_items = ss.top_k();
    EXPECT_EQ(ss_items.size(), K);

    std::unordered_set<uint64_t> ss_top_keys;
    for (const auto& item : ss_items) {
        ss_top_keys.insert(item.key);
    }

    std::size_t true_positives = 0;
    for (uint64_t k : ss_top_keys) {
        if (true_top_keys.contains(k)) {
            true_positives++;
        }
    }

    // 100% precision and 100% recall
    EXPECT_EQ(true_positives, K)
        << "SpaceSaving must achieve 100% precision and recall on Top-100 Zipf";
}

TEST(SpaceSavingTest, ClearResetsState) {
    SpaceSavingHeavyHitters<uint64_t, 10> ss;
    for (uint64_t i = 1; i <= 20; ++i) {
        ss.add(i, 5);
    }
    EXPECT_EQ(ss.top_k().size(), 10ULL);
    ss.clear();
    EXPECT_TRUE(ss.top_k().empty());

    // Re-use after clear
    ss.add(99, 12);
    auto items = ss.top_k();
    ASSERT_EQ(items.size(), 1ULL);
    EXPECT_EQ(items[0].key, 99ULL);
    EXPECT_EQ(items[0].count, 12ULL);
    EXPECT_EQ(items[0].error, 0ULL);
}

// ============================================================================
// Slice 5: Heterogeneous KeyType Support (int, string, string_view, struct)
// ============================================================================

TEST(CountMinSketchTest, SingleThreadedThroughputSmokeTest) {
    // 1. Standard Count-Min Sketch
    {
        CountMinSketch<uint64_t, uint32_t> cms(0.001, 0.01, false);
        constexpr std::size_t N = 1000000;
        auto start = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < N; ++i) {
            cms.add(i, 1);
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        double sec = std::chrono::duration<double>(elapsed).count();
        double mops = (static_cast<double>(N) / sec) / 1e6;
#if defined(NDEBUG)
        EXPECT_GE(mops, 40.0)
            << "Standard CMS update throughput must exceed 40 M ops/s in release mode (measured: "
            << mops << " M ops/s)";
#else
        EXPECT_GT(mops, 1.0);
#endif
    }

    // 2. Conservative Update Count-Min Sketch (Default)
    {
        CountMinSketch<uint64_t, uint32_t> cms(0.001, 0.01, true);
        constexpr std::size_t N = 1000000;
        auto start = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < N; ++i) {
            cms.add(i, 1);
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        double sec = std::chrono::duration<double>(elapsed).count();
        double mops = (static_cast<double>(N) / sec) / 1e6;
#if defined(NDEBUG)
        EXPECT_GE(mops, 40.0)
            << "Conservative Update CMS throughput must exceed 40 M ops/s in release mode "
               "(measured: "
            << mops << " M ops/s)";
#else
        EXPECT_GT(mops, 1.0);
#endif
    }
}

struct CustomPacketHeader {
    uint32_t src_ip{0};
    uint32_t dst_ip{0};
    uint16_t src_port{0};
    uint16_t dst_port{0};
    uint8_t protocol{0};

    bool operator==(const CustomPacketHeader& o) const noexcept {
        return src_ip == o.src_ip && dst_ip == o.dst_ip && src_port == o.src_port &&
               dst_port == o.dst_port && protocol == o.protocol;
    }

    auto operator<=>(const CustomPacketHeader&) const = default;
};

template <> struct std::hash<CustomPacketHeader> {
    std::size_t operator()(const CustomPacketHeader& h) const noexcept {
        return static_cast<std::size_t>(h.src_ip ^ (static_cast<uint64_t>(h.dst_ip) << 32) ^
                                        (static_cast<uint64_t>(h.src_port) << 16) ^ h.dst_port ^
                                        h.protocol);
    }
};

TEST(HeterogeneousTypesTest, CountMinSketchSupportsMultipleTypes) {
    // 1. Signed int
    {
        CountMinSketch<int, uint32_t> cms(0.01, 0.05);
        cms.add(-42, 5);
        cms.add(100, 10);
        EXPECT_GE(cms.estimate(-42), 5U);
        EXPECT_GE(cms.estimate(100), 10U);
        EXPECT_EQ(cms.estimate(0), 0U);
    }

    // 2. std::string
    {
        CountMinSketch<std::string, uint32_t> cms(0.01, 0.05);
        cms.add("user_alpha", 3);
        cms.add("user_beta", 7);
        EXPECT_GE(cms.estimate("user_alpha"), 3U);
        EXPECT_GE(cms.estimate("user_beta"), 7U);
        EXPECT_EQ(cms.estimate("user_gamma"), 0U);
    }

    // 3. std::string_view
    {
        CountMinSketch<std::string_view, uint32_t> cms(0.01, 0.05);
        std::string s1 = "stream_token_1";
        std::string s2 = "stream_token_2";
        cms.add(s1, 15);
        cms.add(s2, 20);
        EXPECT_GE(cms.estimate(s1), 15U);
        EXPECT_GE(cms.estimate(s2), 20U);
    }

    // 4. Custom struct
    {
        CountMinSketch<CustomPacketHeader, uint32_t> cms(0.01, 0.05);
        CustomPacketHeader h1{.src_ip = 0x0a000001,
                              .dst_ip = 0x0a000002,
                              .src_port = 80,
                              .dst_port = 443,
                              .protocol = 6};
        CustomPacketHeader h2{.src_ip = 0x0a000003,
                              .dst_ip = 0x0a000004,
                              .src_port = 53,
                              .dst_port = 53,
                              .protocol = 17};
        cms.add(h1, 42);
        cms.add(h2, 88);
        EXPECT_GE(cms.estimate(h1), 42U);
        EXPECT_GE(cms.estimate(h2), 88U);
    }
}

TEST(HeterogeneousTypesTest, SpaceSavingSupportsMultipleTypes) {
    // 1. std::string
    {
        SpaceSavingHeavyHitters<std::string, 5> ss;
        ss.add("foo", 10);
        ss.add("bar", 25);
        ss.add("baz", 5);

        auto items = ss.top_k();
        ASSERT_EQ(items.size(), 3ULL);
        EXPECT_EQ(items[0].key, "bar");
        EXPECT_EQ(items[0].count, 25ULL);
        EXPECT_EQ(items[1].key, "foo");
        EXPECT_EQ(items[1].count, 10ULL);
        EXPECT_EQ(items[2].key, "baz");
        EXPECT_EQ(items[2].count, 5ULL);
    }

    // 2. Custom struct
    {
        SpaceSavingHeavyHitters<CustomPacketHeader, 5> ss;
        CustomPacketHeader h1{
            .src_ip = 1, .dst_ip = 2, .src_port = 80, .dst_port = 443, .protocol = 6};
        CustomPacketHeader h2{
            .src_ip = 3, .dst_ip = 4, .src_port = 53, .dst_port = 53, .protocol = 17};
        ss.add(h1, 100);
        ss.add(h2, 200);

        auto items = ss.top_k();
        ASSERT_EQ(items.size(), 2ULL);
        EXPECT_EQ(items[0].key, h2);
        EXPECT_EQ(items[0].count, 200ULL);
        EXPECT_EQ(items[1].key, h1);
        EXPECT_EQ(items[1].count, 100ULL);
    }
}

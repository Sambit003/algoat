#include "algoat/sorting/loser_tree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <queue>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace algoat::sorting::testing {
namespace {

// Reference multiway merge using std::priority_queue
template <typename T> std::vector<T> pq_reference_merge(const std::vector<std::vector<T>>& runs) {
    struct Element {
        T val;
        std::size_t stream_idx;
        std::size_t elem_idx;

        bool operator>(const Element& other) const {
            return val > other.val;
        }
    };

    std::priority_queue<Element, std::vector<Element>, std::greater<Element>> pq;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        if (!runs[i].empty()) {
            pq.push(Element{runs[i][0], i, 0});
        }
    }

    std::vector<T> result;
    while (!pq.empty()) {
        auto top = pq.top();
        pq.pop();
        result.push_back(top.val);

        if (top.elem_idx + 1 < runs[top.stream_idx].size()) {
            pq.push(
                Element{runs[top.stream_idx][top.elem_idx + 1], top.stream_idx, top.elem_idx + 1});
        }
    }
    return result;
}

// Non-default-constructible type to verify LoserTree does not require default construction
struct NoDefaultConstructible {
    int value;

    explicit NoDefaultConstructible(int v) : value(v) {}
    NoDefaultConstructible() = delete;

    auto operator<=>(const NoDefaultConstructible& other) const noexcept = default;
};

TEST(LoserTreeTest, NonDefaultConstructibleElements) {
    std::vector<std::vector<NoDefaultConstructible>> runs;
    runs.push_back(
        {NoDefaultConstructible{1}, NoDefaultConstructible{4}, NoDefaultConstructible{7}});
    runs.push_back(
        {NoDefaultConstructible{2}, NoDefaultConstructible{5}, NoDefaultConstructible{8}});
    runs.push_back(
        {NoDefaultConstructible{3}, NoDefaultConstructible{6}, NoDefaultConstructible{9}});

    std::vector<NoDefaultConstructible> out;
    k_way_merge(runs, std::back_inserter(out));

    ASSERT_EQ(out.size(), 9u);
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(out[static_cast<std::size_t>(i)].value, i + 1);
    }
}

TEST(LoserTreeTest, EmptyAndSingleStream) {
    // K = 0
    LoserTree<int> tree0(0);
    EXPECT_TRUE(tree0.empty());
    EXPECT_EQ(tree0.size(), 0u);

    // K = 1
    LoserTree<int> tree1(1);
    EXPECT_TRUE(tree1.empty());
    EXPECT_EQ(tree1.size(), 0u);

    std::vector<int> head1 = {42};
    tree1.initialize(head1);
    EXPECT_FALSE(tree1.empty());
    EXPECT_EQ(tree1.size(), 1u);
    EXPECT_EQ(tree1.top_source(), 0u);
    EXPECT_EQ(tree1.top(), 42);

    tree1.replace_top(100);
    EXPECT_FALSE(tree1.empty());
    EXPECT_EQ(tree1.size(), 1u);
    EXPECT_EQ(tree1.top_source(), 0u);
    EXPECT_EQ(tree1.top(), 100);

    tree1.pop_stream();
    EXPECT_TRUE(tree1.empty());
    EXPECT_EQ(tree1.size(), 0u);
}

TEST(LoserTreeTest, TwoStreamsTournament) {
    LoserTree<int> tree(2);
    std::vector<int> heads = {20, 10};
    tree.initialize(heads);

    EXPECT_FALSE(tree.empty());
    EXPECT_EQ(tree.size(), 2u);
    EXPECT_EQ(tree.top_source(), 1u);
    EXPECT_EQ(tree.top(), 10);

    // Stream 1 gets new value 25
    tree.replace_top(25);
    EXPECT_EQ(tree.top_source(), 0u);
    EXPECT_EQ(tree.top(), 20);

    // Stream 0 gets new value 30
    tree.replace_top(30);
    EXPECT_EQ(tree.top_source(), 1u);
    EXPECT_EQ(tree.top(), 25);

    // Stream 1 is exhausted
    tree.pop_stream();
    EXPECT_FALSE(tree.empty());
    EXPECT_EQ(tree.size(), 1u);
    EXPECT_EQ(tree.top_source(), 0u);
    EXPECT_EQ(tree.top(), 30);

    // Stream 0 is exhausted
    tree.pop_stream();
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.size(), 0u);
}

TEST(LoserTreeTest, NonPowerOfTwoStreams) {
    for (std::size_t K : {3u, 5u, 6u, 7u, 9u, 13u, 17u, 31u}) {
        LoserTree<int> tree(K);
        std::vector<int> heads(K);
        for (std::size_t i = 0; i < K; ++i) {
            heads[i] = static_cast<int>(K - i); // heads: K, K-1, ..., 1
        }
        tree.initialize(heads);

        EXPECT_FALSE(tree.empty());
        EXPECT_EQ(tree.size(), K);
        EXPECT_EQ(tree.top(), 1);
        EXPECT_EQ(tree.top_source(), K - 1);

        // Exhaust each stream in ascending order
        for (std::size_t step = 0; step < K; ++step) {
            EXPECT_FALSE(tree.empty());
            EXPECT_EQ(tree.size(), K - step);
            EXPECT_EQ(tree.top(), static_cast<int>(step + 1));
            tree.pop_stream();
        }
        EXPECT_TRUE(tree.empty());
        EXPECT_EQ(tree.size(), 0u);
    }
}

// Counting comparator to verify strictly <= ceil(log2 K) comparisons per replace_top
struct CountingLess {
    mutable std::size_t count = 0;

    template <typename T> bool operator()(const T& a, const T& b) const {
        ++count;
        return a < b;
    }
};

TEST(LoserTreeTest, StrictComparisonBoundPerReplaceTop) {
    for (std::size_t K :
         {2u, 3u, 4u, 5u, 7u, 8u, 15u, 16u, 31u, 32u, 63u, 64u, 127u, 128u, 256u, 512u, 1024u}) {
        CountingLess comp;
        LoserTree<int, CountingLess> tree(K, comp);

        std::vector<int> heads(K, 0);
        tree.initialize(heads);

        std::size_t max_expected_comparisons =
            static_cast<std::size_t>(std::ceil(std::log2(static_cast<double>(K))));

        std::mt19937_64 rng(1337 + K);
        std::uniform_int_distribution<int> dist(1, 100000);

        // 1. Verify bound when all streams are active
        for (std::size_t iter = 0; iter < 100; ++iter) {
            comp.count = 0;
            int next_val = dist(rng);
            tree.replace_top(next_val);
            EXPECT_LE(comp.count, max_expected_comparisons)
                << "Failed comparison bound for K=" << K << " (all active) on iteration " << iter
                << ": executed " << comp.count << " comparisons, bound is "
                << max_expected_comparisons;
        }

        // 2. Verify bound when streams are progressively depleted
        for (std::size_t depleted = 1; depleted < K; ++depleted) {
            tree.pop_stream();
            if (tree.empty())
                break;
            comp.count = 0;
            tree.replace_top(dist(rng));
            EXPECT_LE(comp.count, max_expected_comparisons)
                << "Failed comparison bound for K=" << K << " with " << depleted
                << " depleted streams"
                << ": executed " << comp.count << " comparisons, bound is "
                << max_expected_comparisons;
        }
    }
}

TEST(LoserTreeTest, CustomGreaterComparator) {
    // Max-tournament using std::ranges::greater
    std::size_t K = 5;
    LoserTree<int, std::ranges::greater> max_tree(K, std::ranges::greater{});
    std::vector<int> heads = {10, 50, 30, 20, 40};
    max_tree.initialize(heads);

    EXPECT_EQ(max_tree.top(), 50);
    EXPECT_EQ(max_tree.top_source(), 1u);

    max_tree.replace_top(15);
    EXPECT_EQ(max_tree.top(), 40);
    EXPECT_EQ(max_tree.top_source(), 4u);
}

TEST(LoserTreeTest, HeavyObjectsZeroDeepCopy) {
    std::vector<std::vector<std::string>> runs = {
        {"alpha", "delta", "golf"}, {"bravo", "echo", "hotel"}, {"charlie", "foxtrot", "india"}};

    std::vector<std::string> out;
    k_way_merge(runs, std::back_inserter(out));

    std::vector<std::string> expected = {"alpha",   "bravo", "charlie", "delta", "echo",
                                         "foxtrot", "golf",  "hotel",   "india"};
    EXPECT_EQ(out, expected);
}

TEST(KWayMergeTest, EmptyAndSingleRange) {
    std::vector<std::vector<int>> empty_runs;
    std::vector<int> out;
    k_way_merge(empty_runs, std::back_inserter(out));
    EXPECT_TRUE(out.empty());

    std::vector<std::vector<int>> single_run = {{1, 3, 5, 7, 9}};
    out.clear();
    k_way_merge(single_run, std::back_inserter(out));
    EXPECT_EQ(out, (std::vector<int>{1, 3, 5, 7, 9}));
}

TEST(KWayMergeTest, MultipleRangesVaryingLengths) {
    std::vector<std::vector<int>> runs = {{1, 5, 9, 13},          {2, 6, 10},         {},
                                          {3, 7, 11, 15, 17, 19}, {4, 8, 12, 14, 16}, {}};

    std::vector<int> out;
    k_way_merge(runs, std::back_inserter(out));

    std::vector<int> expected;
    for (const auto& run : runs) {
        expected.insert(expected.end(), run.begin(), run.end());
    }
    std::sort(expected.begin(), expected.end());

    EXPECT_EQ(out, expected);
}

TEST(KWayMergeTest, VerificationAcrossStreamRangeAndBitLevelMatch) {
    // Verification across K in [2, 1024] boundary and representative sizes
    for (std::size_t K : {2u, 3u, 7u, 16u, 64u, 128u, 1024u}) {
        const std::size_t elements_per_stream = 10000 / K;
        const std::size_t total = K * elements_per_stream;

        std::vector<std::vector<int>> runs(K);
        std::mt19937 rng(42 + static_cast<unsigned>(K));

        for (std::size_t i = 0; i < K; ++i) {
            runs[i].resize(elements_per_stream);
            for (std::size_t j = 0; j < elements_per_stream; ++j) {
                runs[i][j] = static_cast<int>(rng() % 100000);
            }
            std::sort(runs[i].begin(), runs[i].end());
        }

        std::vector<int> merged;
        merged.reserve(total);
        k_way_merge(runs, std::back_inserter(merged));

        // Bit-level match against reference priority queue merge
        std::vector<int> expected_pq = pq_reference_merge(runs);
        ASSERT_EQ(merged, expected_pq) << "Mismatch against std::priority_queue for K=" << K;

        // Bit-level match against std::sort
        std::vector<int> expected_sort;
        for (const auto& r : runs) {
            expected_sort.insert(expected_sort.end(), r.begin(), r.end());
        }
        std::sort(expected_sort.begin(), expected_sort.end());
        ASSERT_EQ(merged, expected_sort) << "Mismatch against std::sort for K=" << K;
    }
}

TEST(KWayMergeTest, StressTenMillionElements) {
    const std::size_t K = 64;
    const std::size_t elements_per_stream = 156250; // 64 * 156,250 = 10,000,000 elements
    const std::size_t total_elements = K * elements_per_stream;

    std::vector<std::vector<int>> runs(K);
    std::mt19937 rng(42);

    for (std::size_t i = 0; i < K; ++i) {
        runs[i].resize(elements_per_stream);
        for (std::size_t j = 0; j < elements_per_stream; ++j) {
            runs[i][j] = static_cast<int>(rng() % 10000000);
        }
        std::sort(runs[i].begin(), runs[i].end());
    }

    std::vector<int> merged;
    merged.reserve(total_elements);
    k_way_merge(runs, std::back_inserter(merged));

    ASSERT_EQ(merged.size(), total_elements);
    EXPECT_TRUE(std::is_sorted(merged.begin(), merged.end()));
}

} // namespace
} // namespace algoat::sorting::testing

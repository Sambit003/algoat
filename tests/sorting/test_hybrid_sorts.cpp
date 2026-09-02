#include "algoat/sorting/blocksort.hpp"
#include "algoat/sorting/introsort.hpp"
#include "algoat/sorting/timsort.hpp"
#include "test_stable_item.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

using namespace algoat::sorting;
using namespace algoat::sorting::testing;

// --- TimSort Tests ---
class TimSortTest : public ::testing::Test {
protected:
    TimSort algo;
};

TEST_F(TimSortTest, IsStable) {
    test_sort_stability(algo);
}

TEST_F(TimSortTest, SawtoothRuns) {
    std::vector<int> data;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 20; ++j) {
            data.push_back(j);
        }
        for (int j = 20; j > 0; --j) {
            data.push_back(j);
        }
    }
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

namespace {
struct ComparisonCounter {
    int value{0};
    static inline int comparisons = 0;

    auto operator<=>(const ComparisonCounter& other) const {
        ++comparisons;
        return value <=> other.value;
    }
    bool operator==(const ComparisonCounter& other) const {
        return value == other.value;
    }
    bool operator<(const ComparisonCounter& other) const {
        ++comparisons;
        return value < other.value;
    }
};
} // namespace

TEST_F(TimSortTest, LinearComparisonsOnPresorted) {
    constexpr int n = 500;
    std::vector<ComparisonCounter> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = ComparisonCounter{i};
    }
    ComparisonCounter::comparisons = 0;
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(),
                               [](const auto& a, const auto& b) { return a.value < b.value; }));
    // A single pre-sorted run takes exactly n - 1 comparisons
    EXPECT_LE(ComparisonCounter::comparisons, n - 1);
}

TEST_F(TimSortTest, LinearComparisonsOnReverseSorted) {
    constexpr int n = 500;
    std::vector<ComparisonCounter> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = ComparisonCounter{n - i};
    }
    ComparisonCounter::comparisons = 0;
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(),
                               [](const auto& a, const auto& b) { return a.value < b.value; }));
    // A single strictly descending run takes n - 1 comparisons to detect and reverses in-place
    EXPECT_LE(ComparisonCounter::comparisons, n - 1);
}

TEST_F(TimSortTest, NaturalRunsAlternating) {
    std::vector<int> data;
    for (int r = 0; r < 5; ++r) {
        for (int i = 0; i < 100; ++i) {
            data.push_back(i * 2);
        }
        for (int i = 100; i > 0; --i) {
            data.push_back(i * 2 - 1);
        }
    }
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST_F(TimSortTest, LargeStabilityDataset) {
    std::vector<StableItem> data;
    data.reserve(5000);
    int index = 0;
    for (int i = 0; i < 50; ++i) {
        for (int k = 100; k >= 0; --k) {
            data.push_back(StableItem{k % 10, index++});
        }
    }
    algo.sort(std::span{data});
    verify_stability(data);
}

TEST_F(TimSortTest, GallopingTriggerLoAndHi) {
    std::vector<int> data;
    // Run 1: 500 small elements
    for (int i = 0; i < 500; ++i) {
        data.push_back(i * 2);
    }
    // Run 2: 50 medium elements
    for (int i = 0; i < 50; ++i) {
        data.push_back(1000 + i);
    }
    // Run 3: 500 large elements
    for (int i = 0; i < 500; ++i) {
        data.push_back(2000 + i * 2);
    }
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST_F(TimSortTest, VariableRunLengthsStackCollapse) {
    std::vector<int> data;
    std::vector<int> lengths = {34, 55, 89, 144, 233, 144, 89, 55, 34};
    int val = 100000;
    for (int len : lengths) {
        for (int i = 0; i < len; ++i) {
            data.push_back(val--);
        }
    }
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

// --- BlockSort Tests ---
class BlockSortTest : public ::testing::Test {
protected:
    BlockSort algo;
};

TEST_F(BlockSortTest, IsStable) {
    test_sort_stability(algo);
}

// --- IntroSort Tests ---
class IntroSortTest : public ::testing::Test {
protected:
    IntroSort algo;
};

TEST_F(IntroSortTest, HandlesMedianOfThreeKillerSequence) {
    int n = 10000;
    std::vector<int> data(n);
    for (int i = 0; i < n / 2; ++i) {
        data[i] = i * 2;
        data[n - 1 - i] = i * 2 + 1;
    }
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

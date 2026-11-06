#include "algoat/sorting/grailsort.hpp"
#include "algoat/sorting/sorting.hpp"
#include "test_stable_item.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace algoat::sorting::testing {
namespace {

// Compile-time constexpr test
constexpr bool test_constexpr_grailsort() {
    std::array<int, 10> arr = {9, 1, 5, 3, 7, 2, 8, 4, 6, 0};
    grailsort(arr.begin(), arr.end());
    for (std::size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

static_assert(test_constexpr_grailsort(), "grailsort must be constexpr evaluatable");
static_assert(SortAlgorithm<GrailSort, int>, "GrailSort must satisfy SortAlgorithm concept");

TEST(GrailSortTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    grailsort(empty.begin(), empty.end());
    EXPECT_TRUE(empty.empty());

    grailsort(std::span<int>{empty});
    EXPECT_TRUE(empty.empty());

    std::vector<int> single = {42};
    grailsort(single.begin(), single.end());
    EXPECT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], 42);

    grailsort(std::span<int>{single});
    EXPECT_EQ(single[0], 42);
}

TEST(GrailSortTest, TwoElements) {
    std::vector<int> sorted_pair = {10, 20};
    grailsort(std::span<int>{sorted_pair});
    EXPECT_EQ(sorted_pair, (std::vector<int>{10, 20}));

    std::vector<int> reversed_pair = {20, 10};
    grailsort(std::span<int>{reversed_pair});
    EXPECT_EQ(reversed_pair, (std::vector<int>{10, 20}));
}

TEST(GrailSortTest, SmallArrayBelowThreshold) {
    for (int size = 3; size < 16; ++size) {
        std::vector<int> data(size);
        std::iota(data.begin(), data.end(), 1);
        std::reverse(data.begin(), data.end());

        grailsort(std::span<int>{data});
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end())) << "Failed for size " << size;
    }
}

TEST(GrailSortTest, DegenerateOneUniqueKey) {
    for (int size : {16, 32, 64, 100, 128}) {
        std::vector<int> data(size, 7);
        grailsort(std::span<int>{data});
        EXPECT_EQ(data, std::vector<int>(size, 7));
    }
}

TEST(GrailSortTest, DegenerateTwoUniqueKeys) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1);
    for (int size : {16, 31, 32, 63, 64, 100, 128}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::stable_sort(expected.begin(), expected.end());

        grailsort(std::span<int>{data});
        EXPECT_EQ(data, expected) << "Failed for size " << size;
    }
}

TEST(GrailSortTest, DegenerateThreeUniqueKeys) {
    std::mt19937 gen(1337);
    std::uniform_int_distribution<int> dist(0, 2);
    for (int size : {16, 25, 49, 64, 81, 100, 128}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::stable_sort(expected.begin(), expected.end());

        grailsort(std::span<int>{data});
        EXPECT_EQ(data, expected) << "Failed for size " << size;
    }
}

TEST(GrailSortTest, MediumSizesRandom) {
    std::mt19937 gen(999);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    for (int size : {16, 20, 32, 45, 64, 90, 128}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::stable_sort(expected.begin(), expected.end());

        grailsort(std::span<int>{data});
        EXPECT_EQ(data, expected) << "Failed for size " << size;
    }
}

TEST(GrailSortTest, LargeArraysRandom) {
    std::mt19937 gen(2026);
    std::uniform_int_distribution<int> dist(-100000, 100000);
    for (int size : {256, 300, 512, 777, 1000, 1024, 2048, 4096}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::stable_sort(expected.begin(), expected.end());

        grailsort(std::span<int>{data});
        EXPECT_EQ(data, expected) << "Failed for size " << size;
    }
}

TEST(GrailSortTest, AlreadySortedAndReverseSorted) {
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<int> sorted_data(size);
        std::iota(sorted_data.begin(), sorted_data.end(), 1);
        auto copy1 = sorted_data;
        grailsort(std::span<int>{copy1});
        EXPECT_EQ(copy1, sorted_data);

        std::vector<int> rev_data = sorted_data;
        std::reverse(rev_data.begin(), rev_data.end());
        grailsort(std::span<int>{rev_data});
        EXPECT_EQ(rev_data, sorted_data);
    }
}

TEST(GrailSortTest, FewUniqueKeysStrategyTwo) {
    std::mt19937 gen(777);
    for (int num_keys : {4, 6, 8, 12}) {
        std::uniform_int_distribution<int> dist(0, num_keys - 1);
        for (int size : {128, 256, 500, 1024}) {
            std::vector<int> data(size);
            for (int& x : data) {
                x = dist(gen);
            }
            std::vector<int> expected = data;
            std::stable_sort(expected.begin(), expected.end());

            grailsort(std::span<int>{data});
            EXPECT_EQ(data, expected) << "Failed for keys=" << num_keys << " size=" << size;
        }
    }
}

TEST(GrailSortTest, CustomComparatorAndProjection) {
    struct Person {
        std::string name;
        int age;
    };

    std::vector<Person> people = {
        {"Alice", 30}, {"Bob", 20}, {"Charlie", 25}, {"David", 20}, {"Eve", 30}};

    grailsort(people.begin(), people.end(), std::ranges::greater{}, &Person::age);
    EXPECT_EQ(people[0].age, 30);
    EXPECT_EQ(people[1].age, 30);
    EXPECT_EQ(people[2].age, 25);
    EXPECT_EQ(people[3].age, 20);
    EXPECT_EQ(people[4].age, 20);

    GrailSort algo;
    std::vector<int> nums = {5, 2, 8, 1, 9};
    algo.sort(std::span<int>{nums}, std::greater<int>{});
    EXPECT_EQ(nums, (std::vector<int>{9, 8, 5, 2, 1}));
}

TEST(GrailSortTest, StabilityVerificationStandard) {
    GrailSort algo;
    test_sort_stability(algo);
}

TEST(GrailSortTest, StabilityVerificationExtensive) {
    std::mt19937 gen(4242);
    for (int num_keys : {2, 3, 5, 10, 20}) {
        std::uniform_int_distribution<int> dist(0, num_keys - 1);
        for (int size : {20, 50, 100, 256, 500, 1000, 1500, 2048}) {
            std::vector<StableItem> items;
            items.reserve(size);
            for (int i = 0; i < size; ++i) {
                items.push_back({dist(gen), i});
            }
            grailsort(std::span<StableItem>{items});
            verify_stability(items);
        }
    }
}

TEST(GrailSortTest, ComparisonCountBounds) {
    struct CountingComp {
        std::size_t count{0};
        bool operator()(int a, int b) {
            ++count;
            return a < b;
        }
    };

    std::mt19937 gen(12345);
    for (int size : {512, 1024, 2048, 4096}) {
        std::uniform_int_distribution<int> dist(0, 1000000);
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }

        CountingComp comp;
        grailsort(data.begin(), data.end(), std::ref(comp), std::identity{});
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

        double max_allowed = 1.15 * size * std::log2(size);
        EXPECT_LE(comp.count, static_cast<std::size_t>(max_allowed))
            << "Comparison count " << comp.count << " exceeded 1.15 * N * log2(N) (" << max_allowed
            << ") for size " << size;
    }
}

TEST(GrailSortTest, ComparisonCountSpecialDistributions) {
    struct CountingComp {
        std::size_t count{0};
        bool operator()(int a, int b) {
            ++count;
            return a < b;
        }
    };

    for (int size : {512, 1024, 2048}) {
        // 1. Already sorted data
        {
            std::vector<int> sorted_data(size);
            std::iota(sorted_data.begin(), sorted_data.end(), 0);
            CountingComp comp;
            grailsort(sorted_data.begin(), sorted_data.end(), std::ref(comp), std::identity{});
            EXPECT_TRUE(std::is_sorted(sorted_data.begin(), sorted_data.end()));
            // In worst case with key extraction + block merge, comparison count remains well within
            // O(N log N)
            double max_allowed = 1.5 * size * std::log2(size);
            EXPECT_LE(comp.count, static_cast<std::size_t>(max_allowed));
        }

        // 2. Reverse sorted data
        {
            std::vector<int> rev_data(size);
            std::iota(rev_data.begin(), rev_data.end(), 0);
            std::reverse(rev_data.begin(), rev_data.end());
            CountingComp comp;
            grailsort(rev_data.begin(), rev_data.end(), std::ref(comp), std::identity{});
            EXPECT_TRUE(std::is_sorted(rev_data.begin(), rev_data.end()));
            double max_allowed = 1.5 * size * std::log2(size);
            EXPECT_LE(comp.count, static_cast<std::size_t>(max_allowed));
        }

        // 3. Few unique keys (e.g. 4 unique keys -> tests Strategy 2 / few keys fallback)
        {
            std::vector<int> few_keys(size);
            for (int i = 0; i < size; ++i) {
                few_keys[i] = i % 4;
            }
            CountingComp comp;
            grailsort(few_keys.begin(), few_keys.end(), std::ref(comp), std::identity{});
            EXPECT_TRUE(std::is_sorted(few_keys.begin(), few_keys.end()));
            double max_allowed = 1.5 * size * std::log2(size);
            EXPECT_LE(comp.count, static_cast<std::size_t>(max_allowed));
        }
    }
}

} // namespace
} // namespace algoat::sorting::testing

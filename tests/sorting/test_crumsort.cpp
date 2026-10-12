#include "algoat/sorting/crumsort.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <concepts>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace algoat::sorting::testing {
namespace {

struct Item {
    int id;
    int priority;

    constexpr auto operator<=>(const Item&) const = default;
};

struct ThrowingComp {
    std::size_t budget;
    mutable std::size_t count{0};

    bool operator()(int a, int b) const {
        if (++count > budget) {
            throw std::runtime_error("ThrowingComp budget exceeded");
        }
        return a < b;
    }
};

TEST(CrumSortTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    crumsort(empty.begin(), empty.end());
    EXPECT_TRUE(empty.empty());

    crumsort(empty);
    EXPECT_TRUE(empty.empty());

    std::vector<int> single = {42};
    crumsort(single.begin(), single.end());
    EXPECT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], 42);

    crumsort(single);
    EXPECT_EQ(single[0], 42);
}

TEST(CrumSortTest, TwoElements) {
    std::vector<int> sorted_pair = {10, 20};
    crumsort(sorted_pair);
    EXPECT_EQ(sorted_pair, (std::vector<int>{10, 20}));

    std::vector<int> reversed_pair = {20, 10};
    crumsort(reversed_pair);
    EXPECT_EQ(reversed_pair, (std::vector<int>{10, 20}));
}

TEST(CrumSortTest, SmallArrayBoundary) {
    for (int size = 3; size <= 100; ++size) {
        std::vector<int> data(size);
        std::iota(data.begin(), data.end(), 1);
        std::reverse(data.begin(), data.end());

        crumsort(data);
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end())) << "Failed for size " << size;
    }
}

TEST(CrumSortTest, MediumArrayNinther) {
    std::mt19937 gen(1337);
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (int size : {128, 256, 512, 1024, 2048}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        crumsort(data);
        EXPECT_EQ(data, expected) << "Failed for size " << size;
    }
}

TEST(CrumSortTest, LargeArrayCbrtPivot) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1000000);

    for (int size : {2049, 4096, 10000, 32768, 65536}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        crumsort(data);
        EXPECT_EQ(data, expected) << "Failed for size " << size;
    }
}

TEST(CrumSortTest, StrictlySortedLinearPath) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    crumsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(CrumSortTest, StrictlyReverseSortedLinearPath) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);
    std::reverse(data.begin(), data.end());

    crumsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(CrumSortTest, PresortedNinetyFivePercent) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    std::mt19937 gen(99);
    std::uniform_int_distribution<std::size_t> idx_dist(0, N - 1);
    for (std::size_t i = 0; i < N / 20; ++i) {
        std::swap(data[idx_dist(gen)], data[idx_dist(gen)]);
    }

    crumsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(CrumSortTest, SingleValueDuplicates) {
    std::vector<int> data(10000, 7);
    crumsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    EXPECT_EQ(data.front(), 7);
    EXPECT_EQ(data.back(), 7);
}

TEST(CrumSortTest, ManyDuplicatesFewUniqueKeys) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    std::mt19937 gen(777);
    std::uniform_int_distribution<int> dist(0, 4);

    for (int& x : data) {
        x = dist(gen);
    }

    std::vector<int> expected = data;
    std::sort(expected.begin(), expected.end());

    crumsort(data);
    EXPECT_EQ(data, expected);
}

TEST(CrumSortTest, PipeOrganDistribution) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N / 2; ++i) {
        data[i] = static_cast<int>(i);
        data[N - 1 - i] = static_cast<int>(i);
    }

    crumsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(CrumSortTest, SawtoothDistribution) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i % 50);
    }

    crumsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(CrumSortTest, CustomComparatorDescending) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    crumsort(data, std::greater<int>{});

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int>{}));
    EXPECT_EQ(data.front(), 9);
    EXPECT_EQ(data.back(), 1);
}

TEST(CrumSortTest, CustomProjection) {
    std::vector<Item> items = {{1, 50}, {2, 10}, {3, 40}, {4, 20}, {5, 30}};

    crumsort(items, std::ranges::less{}, &Item::priority);

    EXPECT_TRUE(std::is_sorted(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.priority < b.priority;
    }));
    EXPECT_EQ(items.front().priority, 10);
    EXPECT_EQ(items.back().priority, 50);
}

TEST(CrumSortTest, NonArithmeticTypesString) {
    std::vector<std::string> words = {"crumsort",  "branchless", "fulcrum", "partition",
                                      "quicksort", "quadsort",   "scandum", "algorithm",
                                      "mergesort", "heapsort"};

    crumsort(words);
    EXPECT_TRUE(std::is_sorted(words.begin(), words.end()));
}

TEST(CrumSortTest, ExceptionSafetyZeroResourceLeaks) {
    constexpr std::size_t N = 500;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 1);
    std::mt19937 gen(42);
    std::shuffle(data.begin(), data.end(), gen);

    for (std::size_t budget : {1, 10, 50, 100, 250, 500}) {
        auto copy = data;
        ThrowingComp comp{.budget = budget};
        try {
            crumsort(copy.begin(), copy.end(), comp);
        } catch (const std::runtime_error& e) {
            EXPECT_STREQ(e.what(), "ThrowingComp budget exceeded");
        }
        EXPECT_EQ(copy.size(), N);
    }
}

struct TrackedResource {
    inline static int live_instances{0};

    int value{0};
    std::shared_ptr<int> resource{};

    TrackedResource() : resource(std::make_shared<int>(0)) {
        ++live_instances;
    }

    explicit TrackedResource(int v) : value(v), resource(std::make_shared<int>(v)) {
        ++live_instances;
    }

    TrackedResource(const TrackedResource& other) : value(other.value), resource(other.resource) {
        ++live_instances;
    }

    TrackedResource& operator=(const TrackedResource& other) {
        value = other.value;
        resource = other.resource;
        return *this;
    }

    TrackedResource(TrackedResource&& other) noexcept
        : value(other.value), resource(std::move(other.resource)) {
        ++live_instances;
    }

    TrackedResource& operator=(TrackedResource&& other) noexcept {
        value = other.value;
        resource = std::move(other.resource);
        return *this;
    }

    ~TrackedResource() {
        --live_instances;
    }
};

struct ThrowingTrackedComp {
    std::size_t budget{0};
    mutable std::size_t count{0};

    bool operator()(const TrackedResource& a, const TrackedResource& b) const {
        if (++count > budget) {
            throw std::runtime_error("ThrowingTrackedComp budget exceeded");
        }
        return a.value < b.value;
    }
};

TEST(CrumSortTest, ExceptionSafetyWithDynamicResourcesZeroLeaks) {
    constexpr std::size_t N = 300;
    for (std::size_t budget : {1, 10, 50, 100, 200}) {
        {
            std::vector<TrackedResource> vec;
            vec.reserve(N);
            for (std::size_t i = 0; i < N; ++i) {
                vec.emplace_back(static_cast<int>(N - i));
            }
            EXPECT_EQ(TrackedResource::live_instances, static_cast<int>(N));

            ThrowingTrackedComp comp{.budget = budget};
            try {
                crumsort(vec.begin(), vec.end(), comp);
            } catch (const std::runtime_error& e) {
                EXPECT_STREQ(e.what(), "ThrowingTrackedComp budget exceeded");
            }

            // All N elements must remain valid and tracked
            EXPECT_EQ(TrackedResource::live_instances, static_cast<int>(N));
            EXPECT_EQ(vec.size(), N);
        }
        // After vector is destroyed, all dynamic resources must be cleanly freed (zero leaks)
        EXPECT_EQ(TrackedResource::live_instances, 0);
    }
}

TEST(CrumSortTest, RandomArraysUpToOneMillion) {
    std::mt19937_64 gen(2026);
    std::uniform_int_distribution<int> dist(-1'000'000, 1'000'000);

    for (std::size_t size : {100UL, 1'000UL, 10'000UL, 100'000UL, 1'000'000UL}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        crumsort(data);
        EXPECT_EQ(data, expected) << "Failed sorting random array of size " << size;
    }
}

TEST(CrumSortTest, AdapterSortAlgorithmConcept) {
    static_assert(SortAlgorithm<CrumSort, int>);
    static_assert(SortAlgorithm<CrumSort, double>);

    CrumSort adapter;
    EXPECT_EQ(adapter.name(), "crumsort");
    EXPECT_EQ(adapter.preferred_min_size(), 0u);

    std::vector<int> data = {9, 3, 7, 1, 5, 2, 8, 6, 4};
    adapter.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    adapter.sort(std::span{data}, std::greater<int>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int>{}));
}

} // namespace
} // namespace algoat::sorting::testing

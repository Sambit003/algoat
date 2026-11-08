#include "algoat/selection/detail/d_ary_heap.hpp"
#include "algoat/selection/dynamic_median.hpp"

#include <algorithm>
#include <functional>
#include <gtest/gtest.h>
#include <queue>
#include <random>
#include <vector>

namespace algoat::selection::test {

TEST(DAryHeapTest, EmptyAndSizeInvariants) {
    detail::d_ary_heap<int, 4, std::less<int>> heap;
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0);
    EXPECT_THROW((void)heap.top(), std::underflow_error);
    EXPECT_THROW(heap.pop(), std::underflow_error);
}

TEST(DAryHeapTest, MaxHeapPushPopMatchesStdPriorityQueue) {
    detail::d_ary_heap<int, 4, std::less<int>> heap;
    std::priority_queue<int> std_pq;

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);

    for (int i = 0; i < 500; ++i) {
        int val = dist(rng);
        heap.push(val);
        std_pq.push(val);
        EXPECT_EQ(heap.size(), std_pq.size());
        EXPECT_EQ(heap.top(), std_pq.top());
    }

    while (!std_pq.empty()) {
        EXPECT_EQ(heap.top(), std_pq.top());
        heap.pop();
        std_pq.pop();
        EXPECT_EQ(heap.size(), std_pq.size());
    }
    EXPECT_TRUE(heap.empty());
}

TEST(DAryHeapTest, MinHeapPushPopMatchesStdPriorityQueue) {
    detail::d_ary_heap<int, 4, std::greater<int>> heap;
    std::priority_queue<int, std::vector<int>, std::greater<int>> std_pq;

    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<int> dist(-10000, 10000);

    for (int i = 0; i < 500; ++i) {
        int val = dist(rng);
        heap.push(val);
        std_pq.push(val);
        EXPECT_EQ(heap.size(), std_pq.size());
        EXPECT_EQ(heap.top(), std_pq.top());
    }

    while (!std_pq.empty()) {
        EXPECT_EQ(heap.top(), std_pq.top());
        heap.pop();
        std_pq.pop();
    }
    EXPECT_TRUE(heap.empty());
}

TEST(DynamicMedianTest, EmptyThrowsException) {
    DynamicMedian<int> dm;
    EXPECT_TRUE(dm.empty());
    EXPECT_EQ(dm.size(), 0);
    EXPECT_THROW((void)dm.median(), std::underflow_error);
}

TEST(DynamicMedianTest, SingleAndSmallStream) {
    DynamicMedian<int> dm;
    dm.push(10);
    EXPECT_FALSE(dm.empty());
    EXPECT_EQ(dm.size(), 1);
    EXPECT_DOUBLE_EQ(dm.median(), 10.0);

    dm.push(20);
    EXPECT_EQ(dm.size(), 2);
    EXPECT_DOUBLE_EQ(dm.median(), 15.0);

    dm.push(30);
    EXPECT_EQ(dm.size(), 3);
    EXPECT_DOUBLE_EQ(dm.median(), 20.0);

    dm.push(5);
    EXPECT_EQ(dm.size(), 4);
    EXPECT_DOUBLE_EQ(dm.median(), 15.0); // sorted: 5, 10, 20, 30 -> (10+20)/2 = 15

    dm.clear();
    EXPECT_TRUE(dm.empty());
    EXPECT_EQ(dm.size(), 0);
    EXPECT_THROW((void)dm.median(), std::underflow_error);
}

TEST(DynamicMedianTest, DuplicateElements) {
    DynamicMedian<int> dm;
    for (int i = 0; i < 100; ++i) {
        dm.push(42);
        EXPECT_DOUBLE_EQ(dm.median(), 42.0);
    }
}

TEST(DynamicMedianTest, CustomComparatorDescending) {
    // With greater comparator, ordering is reversed
    DynamicMedian<int, std::greater<int>> dm(std::greater<int>{});
    dm.push(1);
    dm.push(2);
    dm.push(3);
    dm.push(4);
    dm.push(5);
    EXPECT_DOUBLE_EQ(dm.median(), 3.0);
}

TEST(DynamicMedianTest, StressAgainstStdNthElement) {
    DynamicMedian<int> dm;
    std::vector<int> history;
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<int> dist(-5000, 5000);

    for (int i = 0; i < 2000; ++i) {
        int val = dist(rng);
        dm.push(val);
        history.push_back(val);

        std::vector<int> sorted = history;
        const std::size_t n = sorted.size();
        double expected_median = 0.0;
        if (n % 2 != 0) {
            std::nth_element(sorted.begin(), sorted.begin() + n / 2, sorted.end());
            expected_median = static_cast<double>(sorted[n / 2]);
        } else {
            std::nth_element(sorted.begin(), sorted.begin() + (n / 2 - 1), sorted.end());
            int mid1 = sorted[n / 2 - 1];
            std::nth_element(sorted.begin() + n / 2, sorted.begin() + n / 2, sorted.end());
            int mid2 = sorted[n / 2];
            expected_median = (static_cast<double>(mid1) + static_cast<double>(mid2)) * 0.5;
        }

        EXPECT_DOUBLE_EQ(dm.median(), expected_median);
    }
}

} // namespace algoat::selection::test

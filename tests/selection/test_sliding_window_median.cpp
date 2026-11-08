#include "algoat/selection/dynamic_median.hpp"

#include <algorithm>
#include <deque>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace algoat::selection::test {

TEST(SlidingWindowMedianTest, EmptyAndInvalidWindowSize) {
    EXPECT_THROW(SlidingWindowMedian<int>(0), std::invalid_argument);

    SlidingWindowMedian<int> swm(5);
    EXPECT_TRUE(swm.empty());
    EXPECT_FALSE(swm.full());
    EXPECT_EQ(swm.size(), 0);
    EXPECT_EQ(swm.window_size(), 5);
    EXPECT_THROW((void)swm.median(), std::underflow_error);
    EXPECT_THROW((void)swm.order_statistic(0), std::out_of_range);
}

TEST(SlidingWindowMedianTest, WarmUpPhase) {
    SlidingWindowMedian<int> swm(3);

    swm.push(10);
    EXPECT_FALSE(swm.empty());
    EXPECT_FALSE(swm.full());
    EXPECT_EQ(swm.size(), 1);
    EXPECT_DOUBLE_EQ(swm.median(), 10.0);
    EXPECT_EQ(swm.order_statistic(0), 10);

    swm.push(20);
    EXPECT_FALSE(swm.full());
    EXPECT_EQ(swm.size(), 2);
    EXPECT_DOUBLE_EQ(swm.median(), 15.0);
    EXPECT_EQ(swm.order_statistic(0), 10);
    EXPECT_EQ(swm.order_statistic(1), 20);

    swm.push(30);
    EXPECT_TRUE(swm.full());
    EXPECT_EQ(swm.size(), 3);
    EXPECT_DOUBLE_EQ(swm.median(), 20.0);
    EXPECT_EQ(swm.order_statistic(0), 10);
    EXPECT_EQ(swm.order_statistic(1), 20);
    EXPECT_EQ(swm.order_statistic(2), 30);
}

TEST(SlidingWindowMedianTest, SlidingWindowEvictionLeetCodePattern) {
    // Input: [1, 3, -1, -3, 5, 3, 6, 7], W = 3
    // Windows:
    // [1, 3, -1] -> sorted: [-1, 1, 3] -> median: 1
    // [3, -1, -3] -> sorted: [-3, -1, 3] -> median: -1
    // [-1, -3, 5] -> sorted: [-3, -1, 5] -> median: -1
    // [-3, 5, 3] -> sorted: [-3, 3, 5] -> median: 3
    // [5, 3, 6] -> sorted: [3, 5, 6] -> median: 5
    // [3, 6, 7] -> sorted: [3, 6, 7] -> median: 6
    std::vector<int> nums = {1, 3, -1, -3, 5, 3, 6, 7};
    std::vector<double> expected = {1.0, -1.0, -1.0, 3.0, 5.0, 6.0};

    SlidingWindowMedian<int> swm(3);
    std::vector<double> actual;

    for (int num : nums) {
        swm.push(num);
        if (swm.full()) {
            actual.push_back(swm.median());
        }
    }

    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual[i], expected[i]);
    }
}

TEST(SlidingWindowMedianTest, OrderStatisticRankSelection) {
    const std::size_t W = 5;
    SlidingWindowMedian<int> swm(W);
    std::deque<int> window;

    std::mt19937_64 rng(777);
    std::uniform_int_distribution<int> dist(-100, 100);

    for (int step = 0; step < 200; ++step) {
        int val = dist(rng);
        swm.push(val);
        window.push_back(val);
        if (window.size() > W) {
            window.pop_front();
        }

        std::vector<int> sorted_window(window.begin(), window.end());
        std::sort(sorted_window.begin(), sorted_window.end());

        for (std::size_t r = 0; r < sorted_window.size(); ++r) {
            EXPECT_EQ(swm.order_statistic(r), sorted_window[r]);
        }
        EXPECT_THROW((void)swm.order_statistic(sorted_window.size()), std::out_of_range);
    }
}

TEST(SlidingWindowMedianTest, FreeFunctionSlidingWindowMedian) {
    std::vector<int> data = {1, 3, -1, -3, 5, 3, 6, 7};
    std::vector<double> out;

    algoat::selection::sliding_window_median(data, 3, std::back_inserter(out));
    std::vector<double> expected = {1.0, -1.0, -1.0, 3.0, 5.0, 6.0};
    EXPECT_EQ(out, expected);

    // Empty range
    std::vector<int> empty_data;
    std::vector<double> empty_out;
    algoat::selection::sliding_window_median(empty_data, 3, std::back_inserter(empty_out));
    EXPECT_TRUE(empty_out.empty());

    // Window larger than data
    std::vector<double> short_out;
    algoat::selection::sliding_window_median(data, 100, std::back_inserter(short_out));
    EXPECT_TRUE(short_out.empty());

    // Window size 0
    std::vector<double> zero_w_out;
    algoat::selection::sliding_window_median(data, 0, std::back_inserter(zero_w_out));
    EXPECT_TRUE(zero_w_out.empty());
}

TEST(SlidingWindowMedianTest, DifferentialVerificationAgainstNthElement) {
    const std::vector<std::size_t> test_windows = {3, 4, 7, 16, 31, 64, 128};
    std::mt19937_64 rng(2026);
    std::uniform_int_distribution<int> dist(-100000, 100000);

    for (std::size_t W : test_windows) {
        SlidingWindowMedian<int> swm(W);
        std::deque<int> window;

        for (int i = 0; i < 2000; ++i) {
            int val = dist(rng);
            swm.push(val);
            window.push_back(val);
            if (window.size() > W) {
                window.pop_front();
            }

            if (swm.full()) {
                std::vector<int> sorted(window.begin(), window.end());
                double expected_median = 0.0;
                if (W % 2 != 0) {
                    std::nth_element(sorted.begin(), sorted.begin() + W / 2, sorted.end());
                    expected_median = static_cast<double>(sorted[W / 2]);
                } else {
                    std::nth_element(sorted.begin(), sorted.begin() + (W / 2 - 1), sorted.end());
                    int mid1 = sorted[W / 2 - 1];
                    std::nth_element(sorted.begin() + W / 2, sorted.begin() + W / 2, sorted.end());
                    int mid2 = sorted[W / 2];
                    expected_median = (static_cast<double>(mid1) + static_cast<double>(mid2)) * 0.5;
                }

                EXPECT_DOUBLE_EQ(swm.median(), expected_median);
            }
        }
    }
}

TEST(SlidingWindowMedianTest, HighVolumeEquivalenceLargeWindows) {
    const std::vector<std::size_t> large_windows = {100, 1000, 5000};
    std::mt19937_64 rng(314159);
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    for (std::size_t W : large_windows) {
        SlidingWindowMedian<int> swm(W);
        std::deque<int> window;

        const std::size_t total_steps = W + 100;
        for (std::size_t i = 0; i < total_steps; ++i) {
            int val = dist(rng);
            swm.push(val);
            window.push_back(val);
            if (window.size() > W) {
                window.pop_front();
            }

            if (swm.full() && (i % 10 == 0 || i + 1 == total_steps)) {
                std::vector<int> sorted(window.begin(), window.end());
                double expected_median = 0.0;
                if (W % 2 != 0) {
                    std::nth_element(sorted.begin(), sorted.begin() + W / 2, sorted.end());
                    expected_median = static_cast<double>(sorted[W / 2]);
                } else {
                    std::nth_element(sorted.begin(), sorted.begin() + (W / 2 - 1), sorted.end());
                    int mid1 = sorted[W / 2 - 1];
                    std::nth_element(sorted.begin() + W / 2, sorted.begin() + W / 2, sorted.end());
                    int mid2 = sorted[W / 2];
                    expected_median = (static_cast<double>(mid1) + static_cast<double>(mid2)) * 0.5;
                }
                EXPECT_DOUBLE_EQ(swm.median(), expected_median);
            }
        }
    }
}

} // namespace algoat::selection::test

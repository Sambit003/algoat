#include "algoat/searching/binary_search.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace algoat::searching;

class BinarySearchTest : public ::testing::Test {
protected:
    BinarySearch algo;
};

TEST_F(BinarySearchTest, EmptySpan) {
    std::vector<int> data;
    auto result = algo.search(std::span{data}, 1);
    EXPECT_FALSE(result.has_value());

    std::span<const int> const_span{data.data(), data.size()};
    auto const_result = algo.search(const_span, 1);
    EXPECT_FALSE(const_result.has_value());
}

TEST_F(BinarySearchTest, SingleElement) {
    std::vector<int> data = {42};

    // Matching case
    auto result_match = algo.search(std::span{data}, 42);
    ASSERT_TRUE(result_match.has_value());
    EXPECT_EQ(result_match.value(), 0);

    // Non-matching cases (smaller and larger)
    auto result_smaller = algo.search(std::span{data}, 10);
    EXPECT_FALSE(result_smaller.has_value());

    auto result_larger = algo.search(std::span{data}, 100);
    EXPECT_FALSE(result_larger.has_value());
}

TEST_F(BinarySearchTest, OutOfBoundsTargets) {
    std::vector<int> data = {10, 20, 30, 40, 50};

    // Target smaller than data[0]
    auto result_smaller = algo.search(std::span{data}, 5);
    EXPECT_FALSE(result_smaller.has_value());

    // Target larger than data[N-1]
    auto result_larger = algo.search(std::span{data}, 60);
    EXPECT_FALSE(result_larger.has_value());
}

TEST_F(BinarySearchTest, DuplicateValues) {
    std::vector<int> data = {1, 3, 3, 3, 3, 3, 7};

    auto result = algo.search(std::span{data}, 3);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result.value(), 1);
    EXPECT_LE(result.value(), 5);

    std::vector<int> all_dups = {5, 5, 5, 5};
    auto result_all = algo.search(std::span{all_dups}, 5);
    ASSERT_TRUE(result_all.has_value());
    EXPECT_GE(result_all.value(), 0);
    EXPECT_LE(result_all.value(), 3);
}

TEST_F(BinarySearchTest, EvenAndOddSizes) {
    // Two elements (even)
    std::vector<int> two_elems = {10, 20};
    EXPECT_EQ(algo.search(std::span{two_elems}, 10), 0);
    EXPECT_EQ(algo.search(std::span{two_elems}, 20), 1);
    EXPECT_FALSE(algo.search(std::span{two_elems}, 15).has_value());

    // Three elements (odd)
    std::vector<int> three_elems = {10, 20, 30};
    EXPECT_EQ(algo.search(std::span{three_elems}, 10), 0);
    EXPECT_EQ(algo.search(std::span{three_elems}, 20), 1);
    EXPECT_EQ(algo.search(std::span{three_elems}, 30), 2);
    EXPECT_FALSE(algo.search(std::span{three_elems}, 25).has_value());

    // Four elements (even)
    std::vector<int> four_elems = {10, 20, 30, 40};
    EXPECT_EQ(algo.search(std::span{four_elems}, 10), 0);
    EXPECT_EQ(algo.search(std::span{four_elems}, 20), 1);
    EXPECT_EQ(algo.search(std::span{four_elems}, 30), 2);
    EXPECT_EQ(algo.search(std::span{four_elems}, 40), 3);
    EXPECT_FALSE(algo.search(std::span{four_elems}, 5).has_value());
}

TEST_F(BinarySearchTest, ConstSpan) {
    const std::vector<double> data = {1.1, 2.2, 3.3, 4.4, 5.5};
    std::span<const double> const_span{data.data(), data.size()};

    auto res = algo.search(const_span, 3.3);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 2);
}

TEST_F(BinarySearchTest, NegativeNumbers) {
    // Mixed negative and positive signed integers
    std::vector<int> mixed = {-50, -25, -10, -5, 0, 5, 10, 25, 50};
    EXPECT_EQ(algo.search(std::span{mixed}, -50), 0);
    EXPECT_EQ(algo.search(std::span{mixed}, -10), 2);
    EXPECT_EQ(algo.search(std::span{mixed}, 0), 4);
    EXPECT_EQ(algo.search(std::span{mixed}, 50), 8);
    EXPECT_FALSE(algo.search(std::span{mixed}, -100).has_value());
    EXPECT_FALSE(algo.search(std::span{mixed}, -15).has_value());
    EXPECT_FALSE(algo.search(std::span{mixed}, 100).has_value());

    // All-negative integer array
    std::vector<int> all_neg = {-100, -80, -60, -40, -20};
    EXPECT_EQ(algo.search(std::span{all_neg}, -100), 0);
    EXPECT_EQ(algo.search(std::span{all_neg}, -60), 2);
    EXPECT_EQ(algo.search(std::span{all_neg}, -20), 4);
    EXPECT_FALSE(algo.search(std::span{all_neg}, -150).has_value());
    EXPECT_FALSE(algo.search(std::span{all_neg}, -50).has_value());
    EXPECT_FALSE(algo.search(std::span{all_neg}, 0).has_value());

    // Negative floating point array
    std::vector<double> neg_doubles = {-50.5, -20.25, -5.1, -0.01, 1.25, 3.75};
    EXPECT_EQ(algo.search(std::span{neg_doubles}, -50.5), 0);
    EXPECT_EQ(algo.search(std::span{neg_doubles}, -0.01), 3);
    EXPECT_EQ(algo.search(std::span{neg_doubles}, 3.75), 5);
    EXPECT_FALSE(algo.search(std::span{neg_doubles}, -10.0).has_value());
    EXPECT_FALSE(algo.search(std::span{neg_doubles}, -100.0).has_value());
}

TEST_F(BinarySearchTest, ExhaustiveSizeBounds1To64) {
    for (std::size_t size = 1; size <= 64; ++size) {
        std::vector<int> data(size);
        for (std::size_t i = 0; i < size; ++i) {
            data[i] = static_cast<int>((i + 1) * 2); // 2, 4, 6, ..., 2*size
        }

        std::span<const int> const_span{data.data(), data.size()};

        // 1. Verify exact hits for every element
        for (std::size_t i = 0; i < size; ++i) {
            int target = data[i];

            auto res = algo.search(std::span{data}, target);
            ASSERT_TRUE(res.has_value()) << "Failed hit for size " << size << ", index " << i;
            EXPECT_EQ(res.value(), i) << "Incorrect index for size " << size << ", index " << i;

            auto const_res = algo.search(const_span, target);
            ASSERT_TRUE(const_res.has_value())
                << "Failed const hit for size " << size << ", index " << i;
            EXPECT_EQ(const_res.value(), i)
                << "Incorrect const index for size " << size << ", index " << i;

            auto indexed_res = algo.search_indexed(data, size, target);
            ASSERT_TRUE(indexed_res.has_value())
                << "Failed indexed hit for size " << size << ", index " << i;
            EXPECT_EQ(indexed_res.value(), i)
                << "Incorrect indexed index for size " << size << ", index " << i;
        }

        // 2. Verify misses below minimum and above maximum
        EXPECT_FALSE(algo.search(std::span{data}, 0).has_value())
            << "False positive on < min for size " << size;
        EXPECT_FALSE(algo.search(std::span{data}, static_cast<int>(2 * size + 2)).has_value())
            << "False positive on > max for size " << size;

        // 3. Verify misses for all interstitial odd numbers
        for (std::size_t i = 0; i <= size; ++i) {
            int missing_target = static_cast<int>(2 * i + 1); // 1, 3, 5, ..., 2*size + 1
            auto res = algo.search(std::span{data}, missing_target);
            EXPECT_FALSE(res.has_value()) << "False positive on interstitial value "
                                          << missing_target << " for size " << size;
        }
    }
}

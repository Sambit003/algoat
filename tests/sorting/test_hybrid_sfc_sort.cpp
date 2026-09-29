#include "algoat/algoat.hpp"

#include <algorithm>
#include <complex>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace algoat;
using namespace algoat::numerics;

TEST(HybridSFCSortTest, BasicSortingSmall) {
    std::vector<std::complex<float>> data = {
        {0.9f, 0.9f}, {0.1f, 0.1f}, {0.1f, 0.9f}, {0.9f, 0.1f}};

    algoat::sort(std::span(data), SpaceFillingCurve::Hybrid);

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), HybridCompare{}));
}

TEST(HybridSFCSortTest, BasicSortingLargeRadix) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);

    std::vector<std::complex<float>> data(100000); // 100K points to trigger Radix sort (>=256)
    for (auto& point : data) {
        point = {dist(gen), dist(gen)};
    }

    algoat::sort(std::span(data), SpaceFillingCurve::Hybrid);

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), HybridCompare{}));
}

TEST(HybridSFCSortTest, NegativeCoordinates) {
    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dist(-500.0f, 500.0f);

    std::vector<std::complex<float>> data(1000);
    for (auto& point : data) {
        point = {dist(gen), dist(gen)};
    }

    algoat::sort(std::span(data), SpaceFillingCurve::Hybrid);

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), HybridCompare{}));
}

TEST(HybridSFCSortTest, DefaultSortBackwardCompatibility) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::complex<float>> data1(300);
    std::vector<std::complex<float>> data2(300);
    for (size_t i = 0; i < 300; ++i) {
        float r = dist(gen);
        float i_val = dist(gen);
        data1[i] = {r, i_val};
        data2[i] = {r, i_val};
    }

    // Default call
    algoat::sort(std::span(data1));

    // Explicit morton
    algoat::sort(std::span(data2), SpaceFillingCurve::Morton);

    // They should be identical
    for (size_t i = 0; i < data1.size(); ++i) {
        EXPECT_EQ(data1[i].real(), data2[i].real());
        EXPECT_EQ(data1[i].imag(), data2[i].imag());
    }
}

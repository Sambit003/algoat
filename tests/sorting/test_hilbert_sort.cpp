#include "algoat/numerics/hilbert.hpp"

#include <complex>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace algoat::numerics;

TEST(HilbertSortTest, Hilbert2DEncoding) {
    // Zero coordinates
    uint64_t z = float_to_hilbert2d(0.0f, 0.0f);
    EXPECT_GT(z, 0); // Given positive IEEE 0 is halfway in uint32_t

    // Strict monotonicity inside positive octant
    uint64_t a = float_to_hilbert2d(1.0f, 1.0f);
    uint64_t b = float_to_hilbert2d(2.0f, 2.0f);
    EXPECT_NE(a, b);
}

TEST(HilbertSortTest, Hilbert3DEncoding) {
    uint64_t a = float_to_hilbert3d(1.0f, 1.0f, 1.0f);
    uint64_t b = float_to_hilbert3d(2.0f, 2.0f, 2.0f);
    EXPECT_NE(a, b);
}

TEST(HilbertSortTest, FallbackSmallSort) {
    std::vector<std::complex<float>> pts = {
        {2.0f, 2.0f}, {-1.0f, -1.0f}, {0.0f, 0.0f}, {10.0f, 1.0f}};

    std::vector<std::complex<float>> copy = pts;
    sort_complex_hilbert<float>(std::span{pts});

    // Sort copy with std::sort and compare to ensure Fallback is equivalent to our comparator
    std::sort(copy.begin(), copy.end(), HilbertCompare{});
    for (size_t i = 0; i < pts.size(); ++i) {
        EXPECT_EQ(pts[i], copy[i]);
    }
}

TEST(HilbertSortTest, RadixSortLarge) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    size_t N = 10000;
    std::vector<std::complex<float>> pts(N);
    for (size_t i = 0; i < N; ++i) {
        pts[i] = {dist(gen), dist(gen)};
    }

    std::vector<std::complex<float>> copy = pts;
    sort_complex_hilbert<float>(std::span{pts});
    std::sort(copy.begin(), copy.end(), HilbertCompare{});

    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(pts[i], copy[i]);
    }
}

TEST(HilbertSortTest, IdenticalCoordinates) {
    std::vector<std::complex<float>> pts = {{1.0f, 1.0f}, {1.0f, 1.0f}, {2.0f, 2.0f}, {1.0f, 1.0f}};

    std::vector<std::complex<float>> copy = pts;
    sort_complex_hilbert<float>(std::span{pts});
    std::sort(copy.begin(), copy.end(), HilbertCompare{});

    for (size_t i = 0; i < pts.size(); ++i) {
        EXPECT_EQ(pts[i], copy[i]);
    }
}

TEST(HilbertSortTest, DoublePrecision) {
    std::vector<std::complex<double>> pts = {{2.0, 2.0}, {-1.0, -1.0}, {0.0, 0.0}, {10.0, 1.0}};

    std::vector<std::complex<double>> copy = pts;
    sort_complex_hilbert<double>(std::span{pts});
    std::sort(copy.begin(), copy.end(), HilbertCompare{});

    for (size_t i = 0; i < pts.size(); ++i) {
        EXPECT_EQ(pts[i], copy[i]);
    }
}

TEST(HilbertSortTest, SpatialClustering) {
    // Hilbert curves should preserve spatial locality.
    // Points close in 2D space should be close in the sorted 1D array.
    std::vector<std::complex<float>> pts = {{0.0f, 0.0f},   {0.1f, 0.1f},   {0.0f, 0.1f},
                                            {0.1f, 0.0f},   {10.0f, 10.0f}, {10.1f, 10.1f},
                                            {10.0f, 10.1f}, {10.1f, 10.0f}};

    sort_complex_hilbert<float>(std::span{pts});

    // The first 4 points should be grouped together, and the next 4 grouped together.
    for (int i = 0; i < 4; ++i) {
        EXPECT_LT(std::abs(pts[i].real()), 1.0f);
        EXPECT_LT(std::abs(pts[i].imag()), 1.0f);
    }
    for (int i = 4; i < 8; ++i) {
        EXPECT_GT(pts[i].real(), 9.0f);
        EXPECT_GT(pts[i].imag(), 9.0f);
    }
}

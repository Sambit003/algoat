#include "algoat/sketches/quantile_sketches.hpp"

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace algoat::sketches;

// ============================================================================
// Slice 1: DDSketch Construction, Ingestion, Min/Max/Count, Clear
// ============================================================================

TEST(DDSketchTest, ConstructorAndInitialState) {
    DDSketch sketch(0.01, 2048);
    EXPECT_EQ(sketch.count(), 0ULL);
    EXPECT_TRUE(std::isinf(sketch.min_value()));
    EXPECT_GT(sketch.min_value(), 0.0); // +infinity
    EXPECT_TRUE(std::isinf(sketch.max_value()));
    EXPECT_LT(sketch.max_value(), 0.0); // -infinity
    EXPECT_GT(sketch.memory_usage_bytes(), 0ULL);
}

TEST(DDSketchTest, AddAndSummaryStatistics) {
    DDSketch sketch(0.01, 2048);

    sketch.add(10.0);
    sketch.add(20.0);
    sketch.add(5.0);
    sketch.add(0.0);
    sketch.add(-15.0);

    EXPECT_EQ(sketch.count(), 5ULL);
    EXPECT_DOUBLE_EQ(sketch.min_value(), -15.0);
    EXPECT_DOUBLE_EQ(sketch.max_value(), 20.0);

    sketch.clear();
    EXPECT_EQ(sketch.count(), 0ULL);
    EXPECT_TRUE(std::isinf(sketch.min_value()));
    EXPECT_TRUE(std::isinf(sketch.max_value()));
}

// ============================================================================
// Slice 2: DDSketch Relative-Error Guarantees & Quantiles
// ============================================================================

TEST(DDSketchTest, EmptySketchQuantileIsNaN) {
    DDSketch sketch;
    EXPECT_TRUE(std::isnan(sketch.get_quantile(0.5)));
    EXPECT_TRUE(std::isnan(sketch.get_quantile(0.0)));
    EXPECT_TRUE(std::isnan(sketch.get_quantile(1.0)));
}

TEST(DDSketchTest, BoundaryQuantilesReturnMinMax) {
    DDSketch sketch(0.01, 2048);
    for (int i = 1; i <= 100; ++i) {
        sketch.add(static_cast<double>(i));
    }
    EXPECT_DOUBLE_EQ(sketch.get_quantile(0.0), 1.0);
    EXPECT_DOUBLE_EQ(sketch.get_quantile(1.0), 100.0);
}

TEST(DDSketchTest, RelativeErrorGuaranteePositiveDistribution) {
    constexpr double alpha = 0.01; // 1% relative error guarantee
    DDSketch sketch(alpha, 4096);

    std::mt19937_64 rng(42);
    std::lognormal_distribution<double> dist(3.0, 1.5);

    constexpr size_t N = 50000;
    std::vector<double> values;
    values.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        double v = dist(rng);
        values.push_back(v);
        sketch.add(v);
    }

    std::sort(values.begin(), values.end());

    const std::vector<double> quantiles = {0.001, 0.01, 0.05, 0.1,  0.25, 0.5,
                                           0.75,  0.9,  0.95, 0.99, 0.999};
    for (double q : quantiles) {
        size_t rank = static_cast<size_t>(q * static_cast<double>(N - 1));
        double exact_val = values[rank];
        double est_val = sketch.get_quantile(q);

        double relative_error = std::abs(est_val - exact_val) / exact_val;
        EXPECT_LE(relative_error, alpha)
            << "Failed at quantile " << q << ": exact=" << exact_val << ", est=" << est_val
            << ", relative_error=" << relative_error;
    }
}

TEST(DDSketchTest, HandlesZeroAndNegativeValues) {
    DDSketch sketch(0.01, 2048);

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 50.0);

    constexpr size_t N = 10000;
    std::vector<double> values;
    values.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        double v = dist(rng);
        values.push_back(v);
        sketch.add(v);
    }

    std::sort(values.begin(), values.end());

    EXPECT_DOUBLE_EQ(sketch.get_quantile(0.0), values.front());
    EXPECT_DOUBLE_EQ(sketch.get_quantile(1.0), values.back());

    double med = sketch.get_quantile(0.5);
    EXPECT_NEAR(med, 0.0, 2.0);

    for (double q : {0.05, 0.1, 0.2, 0.8, 0.9, 0.95}) {
        size_t rank = static_cast<size_t>(q * static_cast<double>(N - 1));
        double exact_val = values[rank];
        double est_val = sketch.get_quantile(q);

        double rel_err = std::abs(est_val - exact_val) / std::abs(exact_val);
        EXPECT_LE(rel_err, 0.02) << "Failed at q=" << q << ": exact=" << exact_val
                                 << ", est=" << est_val;
    }
}

// ============================================================================
// Slice 3: DDSketch Bin Collapsing & Merge
// ============================================================================

TEST(DDSketchTest, BinCollapsingUnderLargeDynamicRange) {
    constexpr size_t max_bins = 64;
    DDSketch sketch(0.01, max_bins);

    for (int exp = 0; exp <= 20; ++exp) {
        sketch.add(std::pow(10.0, static_cast<double>(exp)));
    }

    EXPECT_LE(sketch.memory_usage_bytes(), sizeof(DDSketch) + (max_bins * 2 * sizeof(uint64_t)));
    EXPECT_DOUBLE_EQ(sketch.min_value(), 1.0);
    EXPECT_DOUBLE_EQ(sketch.max_value(), 1e20);

    double p99 = sketch.get_quantile(0.99);
    EXPECT_GT(p99, 1e18);
}

TEST(DDSketchTest, MergeEquivalence) {
    DDSketch serial(0.01, 2048);
    DDSketch part1(0.01, 2048);
    DDSketch part2(0.01, 2048);

    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> dist(1.0, 100000.0);

    for (int i = 0; i < 10000; ++i) {
        double v = dist(rng);
        serial.add(v);
        if (i % 2 == 0) {
            part1.add(v);
        } else {
            part2.add(v);
        }
    }

    EXPECT_EQ(serial.count(), 10000ULL);
    EXPECT_EQ(part1.count() + part2.count(), 10000ULL);

    part1.merge(part2);
    EXPECT_EQ(part1.count(), 10000ULL);
    EXPECT_DOUBLE_EQ(part1.min_value(), serial.min_value());
    EXPECT_DOUBLE_EQ(part1.max_value(), serial.max_value());

    for (double q : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
        double q_serial = serial.get_quantile(q);
        double q_merged = part1.get_quantile(q);
        double rel_diff = std::abs(q_serial - q_merged) / q_serial;
        EXPECT_LE(rel_diff, 0.015)
            << "Merge mismatch at q=" << q << ": serial=" << q_serial << ", merged=" << q_merged;
    }
}

// ============================================================================
// Slice 4: TDigest Construction, Ingestion, and Centroid Compaction
// ============================================================================

TEST(TDigestTest, ConstructorAndInitialState) {
    TDigest digest(100.0);
    EXPECT_EQ(digest.centroid_count(), 0ULL);
    EXPECT_GT(digest.memory_usage_bytes(), 0ULL);
    EXPECT_TRUE(std::isnan(digest.get_quantile(0.5)));
    EXPECT_TRUE(std::isnan(digest.get_cdf(50.0)));
}

TEST(TDigestTest, AddAndCentroidCountBounded) {
    constexpr double compression = 100.0;
    TDigest digest(compression);

    for (int i = 1; i <= 10000; ++i) {
        digest.add(static_cast<double>(i));
    }

    digest.compress();
    EXPECT_GT(digest.centroid_count(), 0ULL);
    EXPECT_LE(digest.centroid_count(), static_cast<size_t>(compression * 2.0));
}

TEST(TDigestTest, AddWeighted) {
    TDigest digest(100.0);
    digest.add_weighted(10.0, 5.0);
    digest.add_weighted(20.0, 15.0);

    EXPECT_DOUBLE_EQ(digest.get_quantile(0.0), 10.0);
    EXPECT_DOUBLE_EQ(digest.get_quantile(1.0), 20.0);
}

// ============================================================================
// Slice 5: TDigest Quantile & CDF Queries, Extreme Tail Precision
// ============================================================================

TEST(TDigestTest, BoundaryQuantilesAndCDF) {
    TDigest digest(100.0);
    for (int i = 1; i <= 1000; ++i) {
        digest.add(static_cast<double>(i));
    }

    EXPECT_DOUBLE_EQ(digest.get_quantile(0.0), 1.0);
    EXPECT_DOUBLE_EQ(digest.get_quantile(1.0), 1000.0);
    EXPECT_DOUBLE_EQ(digest.get_cdf(0.0), 0.0);
    EXPECT_DOUBLE_EQ(digest.get_cdf(1.0), 0.0);
    EXPECT_DOUBLE_EQ(digest.get_cdf(1000.0), 1.0);
    EXPECT_DOUBLE_EQ(digest.get_cdf(2000.0), 1.0);
}

TEST(TDigestTest, QuantileAndCDFInvertibility) {
    TDigest digest(100.0);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int i = 0; i < 20000; ++i) {
        digest.add(dist(rng));
    }

    for (double q : {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}) {
        double val = digest.get_quantile(q);
        double est_q = digest.get_cdf(val);
        EXPECT_NEAR(est_q, q, 0.02)
            << "Invertibility failed at q=" << q << ": val=" << val << ", est_q=" << est_q;
    }
}

TEST(TDigestTest, ExtremeTailPrecision) {
    constexpr double compression = 100.0;
    TDigest digest(compression);

    constexpr size_t N = 100000;
    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> values;
    values.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        double v = dist(rng);
        values.push_back(v);
        digest.add(v);
    }

    std::sort(values.begin(), values.end());

    // Acceptance Criteria 2: Tail quantiles q in [0.99, 0.9999] maintain absolute error <= 0.0002
    for (double q : {0.99, 0.999, 0.9999}) {
        size_t rank = static_cast<size_t>(q * static_cast<double>(N - 1));
        double exact_val = values[rank];
        double est_val = digest.get_quantile(q);

        double abs_error = std::abs(est_val - exact_val);
        EXPECT_LE(abs_error, 0.0002)
            << "Upper tail error at q=" << q << ": exact=" << exact_val << ", est=" << est_val;
    }

    // Lower extreme tails
    for (double q : {0.0001, 0.001, 0.01}) {
        size_t rank = static_cast<size_t>(q * static_cast<double>(N - 1));
        double exact_val = values[rank];
        double est_val = digest.get_quantile(q);

        double abs_error = std::abs(est_val - exact_val);
        EXPECT_LE(abs_error, 0.0005)
            << "Lower tail error at q=" << q << ": exact=" << exact_val << ", est=" << est_val;
    }
}

// ============================================================================
// Slice 6: TDigest Merging & Multi-Partition Equivalence
// ============================================================================

TEST(TDigestTest, MergeEquivalence) {
    TDigest serial(100.0);
    TDigest p1(100.0);
    TDigest p2(100.0);

    std::mt19937_64 rng(777);
    std::exponential_distribution<double> dist(0.1);

    for (int i = 0; i < 20000; ++i) {
        double v = dist(rng);
        serial.add(v);
        if (i % 2 == 0) {
            p1.add(v);
        } else {
            p2.add(v);
        }
    }

    p1.merge(p2);

    EXPECT_LE(p1.centroid_count(), 250ULL);
    EXPECT_DOUBLE_EQ(p1.get_quantile(0.0), serial.get_quantile(0.0));
    EXPECT_DOUBLE_EQ(p1.get_quantile(1.0), serial.get_quantile(1.0));

    for (double q : {0.01, 0.1, 0.5, 0.9, 0.99, 0.999}) {
        double val_serial = serial.get_quantile(q);
        double val_merged = p1.get_quantile(q);
        double rel_diff = std::abs(val_serial - val_merged) / val_serial;
        EXPECT_LE(rel_diff, 0.05) << "Merge mismatch at q=" << q << ": serial=" << val_serial
                                  << ", merged=" << val_merged;
    }
}

// ============================================================================
// Slice 7: Acceptance Criteria Tests (Synthetic Distributions & Multi-threading)
// ============================================================================

TEST(AcceptanceCriteriaTest, DDSketchParetoAndLogNormalDynamicRange) {
    constexpr double alpha = 0.01;
    DDSketch sketch(alpha, 4096);

    // Pareto distribution with dynamic range > 10^8
    constexpr size_t N = 100000;
    std::mt19937_64 rng(54321);
    std::uniform_real_distribution<double> u_dist(1e-9, 1.0 - 1e-9);

    std::vector<double> values;
    values.reserve(N);

    constexpr double xm = 1.0;
    constexpr double shape = 1.1;

    for (size_t i = 0; i < N; ++i) {
        double u = u_dist(rng);
        double v = xm / std::pow(1.0 - u, 1.0 / shape);
        values.push_back(v);
        sketch.add(v);
    }

    std::sort(values.begin(), values.end());

    // Add log-normal stream extending lower dynamic range
    std::lognormal_distribution<double> log_dist(-10.0, 2.0);
    for (size_t i = 0; i < N; ++i) {
        double v = log_dist(rng);
        values.push_back(v);
        sketch.add(v);
    }
    std::sort(values.begin(), values.end());

    double dynamic_range = values.back() / values.front();
    EXPECT_GT(dynamic_range, 1e8);

    for (double q : {0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999}) {
        size_t rank = static_cast<size_t>(q * static_cast<double>(values.size() - 1));
        double exact_val = values[rank];
        double est_val = sketch.get_quantile(q);

        double rel_error = std::abs(est_val - exact_val) / exact_val;
        EXPECT_LE(rel_error, alpha)
            << "Relative error exceeded at q=" << q << ": exact=" << exact_val
            << ", est=" << est_val << ", rel_error=" << rel_error;
    }
}

TEST(AcceptanceCriteriaTest, MultiThreadedMergeEquivalence) {
    constexpr size_t total_samples = 200000;
    constexpr size_t num_threads = 8;
    constexpr size_t per_thread = total_samples / num_threads;

    DDSketch serial(0.01, 2048);
    std::vector<double> dataset;
    dataset.reserve(total_samples);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(10.0, 100000.0);

    for (size_t i = 0; i < total_samples; ++i) {
        double v = dist(rng);
        dataset.push_back(v);
        serial.add(v);
    }

    std::vector<DDSketch> thread_sketches(num_threads, DDSketch(0.01, 2048));
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            size_t start = t * per_thread;
            size_t end = (t == num_threads - 1) ? total_samples : start + per_thread;
            for (size_t i = start; i < end; ++i) {
                thread_sketches[t].add(dataset[i]);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    DDSketch merged = thread_sketches[0];
    for (size_t t = 1; t < num_threads; ++t) {
        merged.merge(thread_sketches[t]);
    }

    EXPECT_EQ(merged.count(), total_samples);
    EXPECT_DOUBLE_EQ(merged.min_value(), serial.min_value());
    EXPECT_DOUBLE_EQ(merged.max_value(), serial.max_value());

    for (double q : {0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999}) {
        double serial_q = serial.get_quantile(q);
        double merged_q = merged.get_quantile(q);
        double rel_diff = std::abs(serial_q - merged_q) / serial_q;
        EXPECT_LE(rel_diff, 0.01) << "Threaded merge divergence at q=" << q
                                  << ": serial=" << serial_q << ", merged=" << merged_q;
    }
}

// ============================================================================
// Acceptance Criteria: Single-Threaded Ingestion Throughput Verification
// ============================================================================

TEST(AcceptanceCriteriaTest, IngestionThroughputDDSketchAndTDigest) {
    constexpr size_t num_samples = 5000000; // 5M samples
    std::vector<double> dataset;
    dataset.reserve(num_samples);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(1.0, 100000.0);
    for (size_t i = 0; i < num_samples; ++i) {
        dataset.push_back(dist(rng));
    }

    // 1. Benchmark DDSketch
    {
        DDSketch sketch(0.01, 2048);
        auto start = std::chrono::high_resolution_clock::now();
        for (double v : dataset) {
            sketch.add(v);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double mops = (static_cast<double>(num_samples) / elapsed.count()) / 1e6;
        std::cout << "[ THROUGHPUT ] DDSketch Ingestion: " << mops << " M samples/sec ("
                  << elapsed.count() << "s for " << num_samples << " items)" << std::endl;
        // Smoke test verifying reasonable throughput without flakiness under virtualized CI
        // runners.
        EXPECT_GT(mops, 1.0) << "DDSketch update throughput smoke test failed (measured: " << mops
                             << " M ops/s)";
    }

    // 2. Benchmark t-Digest
    {
        TDigest digest(100.0);
        auto start = std::chrono::high_resolution_clock::now();
        for (double v : dataset) {
            digest.add(v);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double mops = (static_cast<double>(num_samples) / elapsed.count()) / 1e6;
        std::cout << "[ THROUGHPUT ] t-Digest Ingestion: " << mops << " M samples/sec ("
                  << elapsed.count() << "s for " << num_samples << " items)" << std::endl;
        EXPECT_GT(mops, 1.0) << "t-Digest update throughput smoke test failed (measured: " << mops
                             << " M ops/s)";
    }
}

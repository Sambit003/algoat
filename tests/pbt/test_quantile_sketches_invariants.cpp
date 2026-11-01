#include "algoat/sketches/quantile_sketches.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

using namespace algoat::sketches;

// ============================================================================
// 1. DDSketch Invariants: Monotonicity & Boundedness
// ============================================================================

RC_GTEST_PROP(DDSketchPBT, QuantileMonotonicity, (const std::vector<double>& raw_values)) {
    RC_PRE(!raw_values.empty());

    DDSketch sketch(0.01, 2048);
    for (double v : raw_values) {
        if (!std::isnan(v)) {
            sketch.add(v);
        }
    }

    RC_PRE(sketch.count() > 0);

    const auto q1 = *rc::gen::inRange(0, 100);
    const auto q2 = *rc::gen::inRange(q1, 100);

    double quant1 = sketch.get_quantile(static_cast<double>(q1) / 100.0);
    double quant2 = sketch.get_quantile(static_cast<double>(q2) / 100.0);

    RC_ASSERT(quant1 <= quant2);
    RC_ASSERT(quant1 >= sketch.min_value());
    RC_ASSERT(quant2 <= sketch.max_value());
}

RC_GTEST_PROP(DDSketchPBT, MergeInvariants,
              (const std::vector<double>& set_a, const std::vector<double>& set_b)) {
    DDSketch a(0.01, 2048);
    DDSketch b(0.01, 2048);

    for (double v : set_a) {
        if (!std::isnan(v)) {
            a.add(v);
        }
    }
    for (double v : set_b) {
        if (!std::isnan(v)) {
            b.add(v);
        }
    }

    RC_PRE(a.count() > 0 && b.count() > 0);

    DDSketch a_b = a;
    a_b.merge(b);

    DDSketch b_a = b;
    b_a.merge(a);

    RC_ASSERT(a_b.count() == a.count() + b.count());
    RC_ASSERT(b_a.count() == a_b.count());
    RC_ASSERT(a_b.min_value() == std::min(a.min_value(), b.min_value()));
    RC_ASSERT(a_b.max_value() == std::max(a.max_value(), b.max_value()));
    RC_ASSERT(a_b.min_value() == b_a.min_value());
    RC_ASSERT(a_b.max_value() == b_a.max_value());
}

// ============================================================================
// 2. TDigest Invariants: Monotonicity, Boundedness & CDF Range
// ============================================================================

RC_GTEST_PROP(TDigestPBT, QuantileMonotonicity, (const std::vector<double>& raw_values)) {
    RC_PRE(!raw_values.empty());

    TDigest digest(100.0);
    for (double v : raw_values) {
        if (!std::isnan(v)) {
            digest.add(v);
        }
    }

    const auto q1 = *rc::gen::inRange(0, 100);
    const auto q2 = *rc::gen::inRange(q1, 100);

    double quant1 = digest.get_quantile(static_cast<double>(q1) / 100.0);
    double quant2 = digest.get_quantile(static_cast<double>(q2) / 100.0);

    RC_ASSERT(quant1 <= quant2);
}

RC_GTEST_PROP(TDigestPBT, CDFMonotonicityAndRange, (const std::vector<double>& raw_values)) {
    RC_PRE(!raw_values.empty());

    TDigest digest(100.0);
    for (double v : raw_values) {
        if (!std::isnan(v)) {
            digest.add(v);
        }
    }

    const auto idx1 = *rc::gen::inRange<size_t>(0, raw_values.size());
    const auto idx2 = *rc::gen::inRange<size_t>(0, raw_values.size());

    double v1 = raw_values[idx1];
    double v2 = raw_values[idx2];
    if (std::isnan(v1) || std::isnan(v2)) {
        return;
    }
    if (v1 > v2) {
        std::swap(v1, v2);
    }

    double cdf1 = digest.get_cdf(v1);
    double cdf2 = digest.get_cdf(v2);

    RC_ASSERT(cdf1 >= 0.0 && cdf1 <= 1.0);
    RC_ASSERT(cdf2 >= 0.0 && cdf2 <= 1.0);
    RC_ASSERT(cdf1 <= cdf2);
}

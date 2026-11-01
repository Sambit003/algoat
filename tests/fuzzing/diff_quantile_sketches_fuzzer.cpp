#include "algoat/sketches/quantile_sketches.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace algoat::sketches;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < sizeof(double)) {
        return 0;
    }

    DDSketch dd(0.02, 512);
    DDSketch dd2(0.02, 512);
    TDigest td(50.0);
    TDigest td2(50.0);

    const std::size_t chunk_size = sizeof(double);
    size_t ingested = 0;

    for (std::size_t i = 0; i + chunk_size <= size; i += chunk_size) {
        double val = 0.0;
        std::memcpy(&val, data + i, sizeof(double));

        if (!std::isnan(val)) {
            if (ingested % 2 == 0) {
                dd.add(val);
                td.add(val);
            } else {
                dd2.add(val);
                td2.add(val);
            }
            ++ingested;
        }
    }

    if (dd.count() > 0) {
        double p10 = dd.get_quantile(0.1);
        double p50 = dd.get_quantile(0.5);
        double p90 = dd.get_quantile(0.9);

        if (p10 > p50 || p50 > p90) {
            __builtin_trap();
        }
        if (p10 < dd.min_value() || p90 > dd.max_value()) {
            __builtin_trap();
        }

        dd.merge(dd2);
        if (dd.count() != ingested) {
            __builtin_trap();
        }
    }

    if (ingested > 0) {
        double p10 = td.get_quantile(0.1);
        double p50 = td.get_quantile(0.5);
        double p90 = td.get_quantile(0.9);

        if (!std::isnan(p10) && !std::isnan(p50) && !std::isnan(p90)) {
            if (p10 > p50 || p50 > p90) {
                __builtin_trap();
            }
        }

        double cdf_val = td.get_cdf(p50);
        if (!std::isnan(cdf_val)) {
            if (cdf_val < 0.0 || cdf_val > 1.0) {
                __builtin_trap();
            }
        }

        td.merge(td2);
    }

    return 0;
}

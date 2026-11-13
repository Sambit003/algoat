/**
 * @file diff_polygon_boolean_fuzzer.cpp
 * @brief LibFuzzer harness for 2D Polygon Boolean Operations under ASan + UBSan.
 */

#include "algoat/geometry/polygon_boolean.hpp"
#include "algoat/geometry/predicates.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace algoat::geometry;

namespace {

double total_area(const MultiPolygon<Point2D>& mp) {
    double sum = 0.0;
    for (const auto& p : mp) {
        sum += p.area();
    }
    return sum;
}

void verify_multipolygon(const MultiPolygon<Point2D>& mp) {
    for (const auto& p : mp) {
        assert(p.outer_ring.size() >= 3);
        for (const auto& pt : p.outer_ring) {
            assert(std::isfinite(pt.x) && std::isfinite(pt.y));
        }
        assert(detail::ring_signed_area<Point2D>(p.outer_ring) > 0.0);
        for (const auto& h : p.holes) {
            assert(h.size() >= 3);
            for (const auto& pt : h) {
                assert(std::isfinite(pt.x) && std::isfinite(pt.y));
            }
            assert(detail::ring_signed_area<Point2D>(h) < 0.0);
        }
    }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    constexpr std::size_t kMinBytes = 12; // Enough for at least 6 coordinates across 2 polygons
    if (size < kMinBytes) {
        return 0;
    }

    // Read header byte for polygon vertex counts and optional hole toggle
    const std::size_t n_a = 3 + (data[0] % 16);
    const std::size_t n_b = 3 + (data[1] % 16);
    const bool add_hole_a = (data[2] & 1u) != 0;
    const bool snap_grid = (data[2] & 2u) != 0;
    const std::size_t payload_offset = 3;

    const std::size_t avail_pairs = (size - payload_offset) / 2;
    if (avail_pairs < n_a + n_b) {
        return 0;
    }

    auto decode_coord = [&](std::size_t idx) -> double {
        const uint8_t raw = data[payload_offset + idx];
        if (snap_grid) {
            return static_cast<double>(static_cast<int>(raw % 33) - 16);
        }
        return (static_cast<double>(raw) - 128.0) * 0.25;
    };

    BooleanPolygon<Point2D> a;
    for (std::size_t i = 0; i < n_a; ++i) {
        a.outer_ring.push_back({decode_coord(2 * i), decode_coord(2 * i + 1)});
    }
    if (add_hole_a) {
        a.holes.push_back({{-2.0, -2.0}, {2.0, -2.0}, {2.0, 2.0}, {-2.0, 2.0}});
    }

    BooleanPolygon<Point2D> b;
    const std::size_t b_base = 2 * n_a;
    for (std::size_t i = 0; i < n_b; ++i) {
        b.outer_ring.push_back({decode_coord(b_base + 2 * i), decode_coord(b_base + 2 * i + 1)});
    }

    const auto mp_i = polygon_intersection(a, b);
    const auto mp_u = polygon_union(a, b);
    const auto mp_d = polygon_difference(a, b);
    const auto mp_x = polygon_xor(a, b);

    verify_multipolygon(mp_i);
    verify_multipolygon(mp_u);
    verify_multipolygon(mp_d);
    verify_multipolygon(mp_x);

    const double ai = total_area(mp_i);
    const double au = total_area(mp_u);
    const double ad = total_area(mp_d);
    const double ax = total_area(mp_x);

    // Since input rings may self-intersect arbitrarily, compute normalized areas via self-union
    const double norm_a = total_area(polygon_union(a, BooleanPolygon<Point2D>{}));
    const double norm_b = total_area(polygon_union(b, BooleanPolygon<Point2D>{}));
    const double tol = 1e-6 * std::max({1.0, norm_a, norm_b});

    assert(std::abs((au + ai) - (norm_a + norm_b)) <= tol);
    assert(std::abs((ad + ai) - norm_a) <= tol);
    assert(std::abs(ax - (au - ai)) <= tol);

    assert(polygon_difference(a, a).empty());
    assert(polygon_xor(a, a).empty());

    return 0;
}

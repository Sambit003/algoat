/**
 * @file diff_bentley_ottmann_fuzzer.cpp
 * @brief Differential LibFuzzer harness for Bentley-Ottmann segment intersection under ASan+UBSan.
 */

#include "algoat/geometry/bentley_ottmann.hpp"
#include "algoat/geometry/predicates.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

using namespace algoat::geometry;

namespace {

[[nodiscard]] bool pt_less(const Point2D& a, const Point2D& b) noexcept {
    return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
}

[[nodiscard]] bool in_box(const Point2D& a, const Point2D& b, const Point2D& q) noexcept {
    return q.x >= std::min(a.x, b.x) && q.x <= std::max(a.x, b.x) && q.y >= std::min(a.y, b.y) &&
           q.y <= std::max(a.y, b.y);
}

[[nodiscard]] bool independent_pair_intersects(const Point2D& p0, const Point2D& p1,
                                               const Point2D& q0, const Point2D& q1) noexcept {
    if (std::max(p0.x, p1.x) < std::min(q0.x, q1.x) ||
        std::max(q0.x, q1.x) < std::min(p0.x, p1.x) ||
        std::max(p0.y, p1.y) < std::min(q0.y, q1.y) ||
        std::max(q0.y, q1.y) < std::min(p0.y, p1.y)) {
        return false;
    }
    const bool p_pt = (p0.x == p1.x && p0.y == p1.y);
    const bool q_pt = (q0.x == q1.x && q0.y == q1.y);
    if (p_pt && q_pt) {
        return p0.x == q0.x && p0.y == q0.y;
    }
    if (p_pt) {
        return orient2d(q0, q1, p0) == Orientation::Collinear && in_box(q0, q1, p0);
    }
    if (q_pt) {
        return orient2d(p0, p1, q0) == Orientation::Collinear && in_box(p0, p1, q0);
    }

    const Orientation o1 = orient2d(p0, p1, q0);
    const Orientation o2 = orient2d(p0, p1, q1);
    const Orientation o3 = orient2d(q0, q1, p0);
    const Orientation o4 = orient2d(q0, q1, p1);

    if (o1 == Orientation::Collinear && o2 == Orientation::Collinear &&
        o3 == Orientation::Collinear && o4 == Orientation::Collinear) {
        const Point2D a_lo = pt_less(p0, p1) ? p0 : p1;
        const Point2D a_hi = pt_less(p0, p1) ? p1 : p0;
        const Point2D b_lo = pt_less(q0, q1) ? q0 : q1;
        const Point2D b_hi = pt_less(q0, q1) ? q1 : q0;
        const Point2D lo = pt_less(a_lo, b_lo) ? b_lo : a_lo;
        const Point2D hi = pt_less(a_hi, b_hi) ? a_hi : b_hi;
        return !pt_less(hi, lo);
    }
    if ((o1 == Orientation::Collinear && in_box(p0, p1, q0)) ||
        (o2 == Orientation::Collinear && in_box(p0, p1, q1)) ||
        (o3 == Orientation::Collinear && in_box(q0, q1, p0)) ||
        (o4 == Orientation::Collinear && in_box(q0, q1, p1))) {
        return true;
    }
    return o1 != o2 && o3 != o4;
}

template <typename T>
std::vector<std::pair<std::size_t, std::size_t>>
brute_force_pairs(std::span<const LineSegment<T>> segments) {
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    const std::size_t n = segments.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point2D p0{static_cast<double>(segments[i].start[0]),
                         static_cast<double>(segments[i].start[1])};
        const Point2D p1{static_cast<double>(segments[i].end[0]),
                         static_cast<double>(segments[i].end[1])};
        if (!std::isfinite(p0.x) || !std::isfinite(p0.y) || !std::isfinite(p1.x) ||
            !std::isfinite(p1.y)) {
            continue;
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            const Point2D q0{static_cast<double>(segments[j].start[0]),
                             static_cast<double>(segments[j].start[1])};
            const Point2D q1{static_cast<double>(segments[j].end[0]),
                             static_cast<double>(segments[j].end[1])};
            if (!std::isfinite(q0.x) || !std::isfinite(q0.y) || !std::isfinite(q1.x) ||
                !std::isfinite(q1.y)) {
                continue;
            }
            if (independent_pair_intersects(p0, p1, q0, q1)) {
                pairs.emplace_back(i, j);
            }
        }
    }
    return pairs;
}

template <typename T> void verify_against_oracle(std::span<const LineSegment<T>> segs) {
    const auto got = bentley_ottmann(segs);
    const auto exp_pairs = brute_force_pairs(segs);
    const std::size_t cnt = count_intersections(segs);
    if (got.size() != exp_pairs.size() || cnt != exp_pairs.size()) {
        __builtin_trap();
    }
    for (std::size_t k = 0; k < exp_pairs.size(); ++k) {
        if (got[k].segment1_index != exp_pairs[k].first ||
            got[k].segment2_index != exp_pairs[k].second ||
            !std::isfinite(static_cast<double>(got[k].point[0])) ||
            !std::isfinite(static_cast<double>(got[k].point[1]))) {
            __builtin_trap();
        }
    }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < 1 + 4 * sizeof(float)) {
        return 0;
    }

    const uint8_t mode = data[0];
    const uint8_t* payload = data + 1;
    const std::size_t payload_bytes = size - 1;

    if ((mode & 1u) != 0) {
        const std::size_t count = std::min<std::size_t>(payload_bytes / 4, 128);
        std::vector<LineSegment<double>> segs;
        segs.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const double x1 = static_cast<double>(static_cast<int8_t>(payload[4 * i]) % 7);
            const double y1 = static_cast<double>(static_cast<int8_t>(payload[4 * i + 1]) % 7);
            const double x2 = static_cast<double>(static_cast<int8_t>(payload[4 * i + 2]) % 7);
            const double y2 = static_cast<double>(static_cast<int8_t>(payload[4 * i + 3]) % 7);
            segs.push_back({{x1, y1}, {x2, y2}});
        }
        verify_against_oracle<double>(segs);
        return 0;
    }

    constexpr std::size_t kSegBytes = 4 * sizeof(float);
    const std::size_t count = std::min<std::size_t>(payload_bytes / kSegBytes, 128);
    std::vector<LineSegment<float>> segs_f;
    std::vector<LineSegment<double>> segs_d;
    segs_f.reserve(count);
    segs_d.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        float raw[4];
        std::memcpy(raw, payload + i * kSegBytes, kSegBytes);
        segs_f.push_back({{raw[0], raw[1]}, {raw[2], raw[3]}});
        segs_d.push_back({{static_cast<double>(raw[0]), static_cast<double>(raw[1])},
                          {static_cast<double>(raw[2]), static_cast<double>(raw[3])}});
    }

    verify_against_oracle<float>(segs_f);
    verify_against_oracle<double>(segs_d);
    return 0;
}

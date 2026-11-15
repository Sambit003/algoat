/**
 * @file diff_rstar_tree_fuzzer.cpp
 * @brief Differential Fuzzer for RStarTree comparing dynamic/bulk-loaded operations
 *        against brute force under ASan + UBSan.
 */

#include "algoat/geometry/rstar_tree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

using namespace algoat::geometry;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    constexpr std::size_t kBoxBytes = 4 * sizeof(float);
    constexpr std::size_t kHeaderBytes = kBoxBytes + 2 * sizeof(float) + 1;
    if (size < kHeaderBytes + kBoxBytes) {
        return 0;
    }

    const uint8_t flags = data[0];
    const bool use_bulk_load = (flags & 1u) != 0;
    const std::size_t k_req = 1 + (flags % 8);

    float qraw[4];
    std::memcpy(qraw, data + 1, kBoxBytes);
    for (float v : qraw) {
        if (!std::isfinite(v)) {
            return 0;
        }
    }
    const BoundingBox<float, 2> query{{std::min(qraw[0], qraw[2]), std::min(qraw[1], qraw[3])},
                                      {std::max(qraw[0], qraw[2]), std::max(qraw[1], qraw[3])}};

    float qpt_raw[2];
    std::memcpy(qpt_raw, data + 1 + kBoxBytes, 2 * sizeof(float));
    if (!std::isfinite(qpt_raw[0]) || !std::isfinite(qpt_raw[1])) {
        return 0;
    }
    const std::array<float, 2> qpt{qpt_raw[0], qpt_raw[1]};

    const std::size_t remaining = size - kHeaderBytes;
    const std::size_t num_boxes = std::min<std::size_t>(remaining / kBoxBytes, 128);

    std::vector<std::pair<BoundingBox<float, 2>, std::size_t>> items;
    items.reserve(num_boxes);
    for (std::size_t i = 0; i < num_boxes; ++i) {
        float b[4];
        std::memcpy(b, data + kHeaderBytes + i * kBoxBytes, kBoxBytes);
        bool ok = true;
        for (float v : b) {
            if (!std::isfinite(v)) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            return 0;
        }
        BoundingBox<float, 2> box{{std::min(b[0], b[2]), std::min(b[1], b[3])},
                                  {std::max(b[0], b[2]), std::max(b[1], b[3])}};
        items.emplace_back(box, i);
    }

    RStarTree<std::size_t, float, 2, 4> tree;
    if (use_bulk_load) {
        tree.bulk_load(items);
    } else {
        for (const auto& [box, idx] : items) {
            tree.insert(box, idx);
        }
    }

    if (tree.size() != items.size()) {
        __builtin_trap();
    }

    // Delete first quarter of entries to fuzz CondenseTree and orphan reinsertion
    std::vector<bool> alive(items.size(), true);
    for (std::size_t i = 0; i < items.size() / 4; ++i) {
        if (!tree.remove(items[i].first, items[i].second)) {
            __builtin_trap();
        }
        alive[i] = false;
    }

    auto actual = tree.search_intersecting(query);
    std::sort(actual.begin(), actual.end());

    std::vector<std::size_t> expected;
    std::vector<double> bf_dists;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (!alive[i]) {
            continue;
        }
        const auto& b = items[i].first;
        if (b.intersects(query)) {
            expected.push_back(i);
        }
        double d2 = 0.0;
        for (std::size_t d = 0; d < 2; ++d) {
            double gap = 0.0;
            if (qpt[d] < b.min_corner[d]) {
                gap = static_cast<double>(b.min_corner[d]) - static_cast<double>(qpt[d]);
            } else if (qpt[d] > b.max_corner[d]) {
                gap = static_cast<double>(qpt[d]) - static_cast<double>(b.max_corner[d]);
            }
            d2 += gap * gap;
        }
        bf_dists.push_back(std::sqrt(d2));
    }
    std::sort(expected.begin(), expected.end());
    if (actual != expected) {
        __builtin_trap();
    }

    const std::size_t k = std::min(k_req, bf_dists.size());
    auto knn = tree.knn_search(qpt, k);
    if (knn.size() != k) {
        __builtin_trap();
    }
    std::sort(bf_dists.begin(), bf_dists.end());
    for (std::size_t i = 0; i < k; ++i) {
        const double tol = 1e-9 * std::max(1.0, bf_dists[i]);
        if (std::abs(knn[i].second - bf_dists[i]) > tol || !std::isfinite(knn[i].second)) {
            __builtin_trap();
        }
    }

    return 0;
}

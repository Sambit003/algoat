/**
 * @file pgm_index.hpp
 * @brief Piecewise Geometric Model (PGM-index) learned search index.
 *
 * Implements the learned index data structure described by Ferragina & Vinciguerra (PVLDB 2020),
 * providing O(1) model prediction and local O(log Epsilon) search with provable worst-case
 * bounds.
 */

#pragma once

#include "algoat/searching/branchless_binary_search.hpp"
#include "algoat/searching/detail/optimal_pla.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::searching {

/**
 * @class PGMIndex
 * @brief Piecewise Geometric Model learned index with provable worst-case error bounds.
 *
 * Approximates the cumulative distribution function (CDF) of sorted keys via an optimal
 * multi-level hierarchy of piecewise linear models bounded by maximum error Epsilon.
 *
 * @tparam KeyType Key element type.
 * @tparam Epsilon Maximum prediction error bound on leaf level.
 * @tparam EpsilonRecursive Maximum prediction error bound on recursive internal levels.
 */
template <typename KeyType, std::size_t Epsilon = 64, std::size_t EpsilonRecursive = 16>
class PGMIndex {
public:
    using key_type = KeyType;
    static constexpr std::size_t epsilon = Epsilon;

    struct Segment {
        KeyType key{};
        float slope{0.0f};
        int32_t intercept{0};
    };

    PGMIndex() = default;

    /**
     * @brief Constructs a multi-level PGM-Index from a random-access range in O(N) time.
     * @param data Range of sorted elements.
     */
    template <std::ranges::random_access_range R>
        requires std::same_as<std::ranges::range_value_t<R>, KeyType>
    explicit PGMIndex(R&& data) {
        build(std::span<const KeyType>{std::ranges::data(data), std::ranges::size(data)});
    }

    /**
     * @brief Builds the optimal multi-level piecewise linear model over sorted data.
     * @param data Sorted view of keys.
     */
    void build(std::span<const KeyType> data) {
        for (auto& lvl : levels_) {
            lvl.clear();
        }
        n_ = data.size();
        levels_count_ = 0;
        first_key_ = data.empty() ? KeyType{} : data.front();

        if (data.empty()) {
            return;
        }

        static constexpr KeyType sentinel = std::numeric_limits<KeyType>::has_infinity
                                                ? std::numeric_limits<KeyType>::infinity()
                                                : std::numeric_limits<KeyType>::max();

        // Build leaf Level 0
        detail::make_segmentation(
            n_, Epsilon, [&](std::size_t i) { return data[i]; },
            [&](const auto& cs) {
                auto [slope, intercept] = cs.get_floating_point_segment();
                levels_[0].push_back(Segment{cs.get_first_x(), slope, intercept});
            });
        levels_[0].push_back(Segment{sentinel, 0.0f, static_cast<int32_t>(n_)});
        levels_count_ = 1;

        // Build recursive internal levels (up to 4 levels total)
        for (std::size_t l = 1; l < 4; ++l) {
            std::size_t num_segs = levels_[l - 1].size() - 1;
            if (num_segs <= 1) {
                break;
            }

            detail::make_segmentation(
                num_segs, EpsilonRecursive, [&](std::size_t i) { return levels_[l - 1][i].key; },
                [&](const auto& cs) {
                    auto [slope, intercept] = cs.get_floating_point_segment();
                    levels_[l].push_back(Segment{cs.get_first_x(), slope, intercept});
                });
            levels_[l].push_back(Segment{sentinel, 0.0f, static_cast<int32_t>(num_segs)});
            ++levels_count_;
        }
    }

    /**
     * @brief Returns an index range [lower, upper] of size <= 2 * Epsilon guaranteed to
     * contain the target key if it exists.
     * @param target Search key.
     * @return Pair [lower, upper] indices in data.
     */
    [[nodiscard]] std::pair<std::size_t, std::size_t>
    search_range(const KeyType& target) const noexcept {
        if (n_ == 0 || levels_[0].empty()) {
            return {0, 0};
        }

        auto it = find_leaf_segment(target);
        std::size_t pos = std::min<std::size_t>(predict_position(*it, target),
                                                static_cast<std::size_t>(std::next(it)->intercept));

        std::size_t lo = pos <= Epsilon ? 0 : pos - Epsilon;
        std::size_t max_idx = n_ > 0 ? n_ - 1 : 0;
        std::size_t hi = std::min(max_idx, lo + 2 * Epsilon);

        if (lo > max_idx) {
            lo = max_idx;
        }
        if (hi > max_idx) {
            hi = max_idx;
        }
        if (lo > hi) {
            lo = hi;
        }
        return {lo, hi};
    }

    /**
     * @brief Direct lower_bound query evaluating model and localized search in O(log Epsilon).
     * @param data Sorted range of keys.
     * @param target Key to locate.
     * @return 0-based index of first element >= target, or data.size() if target > all keys.
     */
    [[nodiscard]] std::size_t lower_bound(std::span<const KeyType> data,
                                          const KeyType& target) const noexcept {
        if (data.empty() || n_ == 0) {
            return 0;
        }
        if (target <= data.front()) {
            return 0;
        }
        if (target > data.back()) {
            return data.size();
        }

        auto it = find_leaf_segment(target);
        std::size_t pos = std::min<std::size_t>(predict_position(*it, target),
                                                static_cast<std::size_t>(std::next(it)->intercept));

        std::size_t lo = pos <= Epsilon ? 0 : pos - Epsilon;
        std::size_t hi = (pos + Epsilon + 2 >= n_) ? n_ : (pos + Epsilon + 2);

        auto res = branchless_lower_bound(data.begin() + lo, data.begin() + hi, target);
        return static_cast<std::size_t>(std::distance(data.begin(), res));
    }

    [[nodiscard]] std::size_t segments_count() const noexcept {
        std::size_t count = 0;
        for (std::size_t l = 0; l < levels_count_; ++l) {
            if (!levels_[l].empty()) {
                count += (levels_[l].size() - 1);
            }
        }
        return count;
    }

    [[nodiscard]] std::size_t size_in_bytes() const noexcept {
        std::size_t total_segs = 0;
        for (std::size_t l = 0; l < levels_count_; ++l) {
            total_segs += levels_[l].size();
        }
        return sizeof(PGMIndex) + total_segs * sizeof(Segment);
    }

private:
    [[nodiscard]] static std::size_t predict_position(const Segment& seg,
                                                      const KeyType& target) noexcept {
        double delta = 0.0;
        if constexpr (std::is_integral_v<KeyType> && sizeof(KeyType) >= 8) {
#if defined(__SIZEOF_INT128__)
            __int128 diff = static_cast<__int128>(target) - static_cast<__int128>(seg.key);
            delta = static_cast<double>(diff);
#else
            delta = static_cast<double>(target) - static_cast<double>(seg.key);
#endif
        } else {
            delta = static_cast<double>(target) - static_cast<double>(seg.key);
        }
        auto pred = static_cast<int64_t>(std::round(seg.slope * delta)) + seg.intercept;
        return pred < 0 ? 0 : static_cast<std::size_t>(pred);
    }

    [[nodiscard]] auto find_leaf_segment(const KeyType& target) const noexcept {
        auto k = std::max(first_key_, target);
        std::size_t root_level = levels_count_ - 1;

        auto it = std::upper_bound(levels_[root_level].begin(), levels_[root_level].end() - 1, k,
                                   [](const KeyType& target_val, const Segment& s) noexcept {
                                       return target_val < s.key;
                                   });
        if (it != levels_[root_level].begin()) {
            --it;
        }

        for (int l = static_cast<int>(levels_count_) - 2; l >= 0; --l) {
            auto level_begin = levels_[l].begin();
            auto pos = std::min<std::size_t>(predict_position(*it, k),
                                             static_cast<std::size_t>(std::next(it)->intercept));
            auto lo =
                level_begin + (pos <= (EpsilonRecursive + 1) ? 0 : pos - (EpsilonRecursive + 1));
            auto level_size = levels_[l].size() - 1;
            if (static_cast<std::size_t>(lo - level_begin) >= level_size) {
                lo = level_begin + (level_size > 0 ? level_size - 1 : 0);
            }
            // Safely scan without ever advancing onto the sentinel
            for (; lo < levels_[l].end() - 2 && std::next(lo)->key <= k; ++lo) {
                continue;
            }
            it = lo;
        }

        return it;
    }

    std::vector<Segment> levels_[4]; // Recursive layers of linear segments
    std::size_t n_{0};
    std::size_t levels_count_{0};
    KeyType first_key_{};
};

} // namespace algoat::searching

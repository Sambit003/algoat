/**
 * @file fractional_cascading.hpp
 * @brief Fractional Cascading multi-list search engine.
 *
 * Implements the seminal data structuring technique of Bernard Chazelle & Leonidas J. Guibas
 * (Algorithmica 1986). Enables simultaneous lower_bound / predecessor searches across K sorted
 * lists in O(log N + K) time and O(N) space.
 */

#pragma once

#include "algoat/searching/branchless_binary_search.hpp"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::searching {

/**
 * @class FractionalCascading
 * @brief Multi-list search engine achieving O(log N + K) queries across K sorted lists.
 *
 * Given K sorted lists L_1, L_2, ..., L_K, independent binary searches require O(K log N) time.
 * FractionalCascading augments each list bottom-up by sampling every second element (alpha = 1/2)
 * from the list below, creating direct downward bridge pointers. A single initial binary search
 * on the top augmented list M_1 followed by O(1) downward bridge transitions locates the lower
 * bound in all K lists in O(log N + K) time.
 *
 * Total augmented memory footprint is strictly bounded by < 2 * sum(|L_i|) *
 * sizeof(AugmentedElement).
 *
 * @tparam T Element value type.
 * @tparam Compare Strict weak ordering comparator callable.
 */
template <typename T, typename Compare = std::ranges::less> class FractionalCascading {
public:
    using value_type = T;

    FractionalCascading() = default;

    /**
     * @brief Constructs the augmented fractional cascade from K sorted lists.
     *
     * @tparam RangeOfRanges A range of sorted ranges convertible to T.
     * @param lists The input collection of K sorted lists.
     * @param comp Strict weak ordering comparator.
     */
    template <std::ranges::input_range RangeOfRanges>
    explicit FractionalCascading(RangeOfRanges&& lists, Compare comp = Compare{})
        : k_(0), comp_(std::move(comp)) {
        // Materialize input into vector of vectors
        std::vector<std::vector<T>> materialized_lists;
        std::size_t total_input_size = 0;
        for (auto&& inner : lists) {
            materialized_lists.emplace_back(std::ranges::begin(inner), std::ranges::end(inner));
            total_input_size += materialized_lists.back().size();
        }

        k_ = materialized_lists.size();
        if (k_ == 0 || total_input_size == 0) {
            if (k_ > 0) {
                augmented_lists_.resize(k_);
                aug_data_.resize(k_, nullptr);
                aug_sizes_.resize(k_, 0);
            }
            return;
        }

        augmented_lists_.resize(k_);
        aug_data_.resize(k_, nullptr);
        aug_sizes_.resize(k_, 0);

        // Base case: construct bottom level M_{k-1}
        {
            const auto& last_list = materialized_lists[k_ - 1];
            auto& last_aug = augmented_lists_[k_ - 1];
            last_aug.reserve(last_list.size() + 1);

            for (std::size_t j = 0; j < last_list.size(); ++j) {
                last_aug.push_back(
                    AugmentedElement{.value = last_list[j], .orig_idx = j, .bridge_idx = 0});
            }

            last_aug.push_back(
                AugmentedElement{.value = make_sentinel_val(materialized_lists, k_ - 1),
                                 .orig_idx = last_list.size(),
                                 .bridge_idx = 0});
        }

        // Inductive step: construct levels bottom-up from k-2 down to 0
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(k_) - 2; i >= 0; --i) {
            const std::size_t cur_idx = static_cast<std::size_t>(i);
            const auto& cur_list = materialized_lists[cur_idx];
            const auto& next_aug = augmented_lists_[cur_idx + 1];
            const std::size_t next_real_count = next_aug.size() - 1;

            // Sample odd indices from M_{i+1} (sampling factor alpha = 1/2)
            struct SampledItem {
                T value;
                std::size_t bridge_idx;
            };

            std::vector<SampledItem> sample;
            sample.reserve(next_real_count / 2);
            for (std::size_t j = 1; j < next_real_count; j += 2) {
                sample.push_back(SampledItem{.value = next_aug[j].value, .bridge_idx = j});
            }

            // Stably merge cur_list and sample into M_i
            auto& cur_aug = augmented_lists_[cur_idx];
            cur_aug.reserve(cur_list.size() + sample.size() + 1);

            std::size_t list_idx = 0;
            std::size_t sample_idx = 0;

            while (list_idx < cur_list.size() && sample_idx < sample.size()) {
                if (!comp_(sample[sample_idx].value, cur_list[list_idx])) {
                    cur_aug.push_back(AugmentedElement{
                        .value = cur_list[list_idx], .orig_idx = list_idx, .bridge_idx = 0});
                    ++list_idx;
                } else {
                    cur_aug.push_back(
                        AugmentedElement{.value = sample[sample_idx].value,
                                         .orig_idx = list_idx,
                                         .bridge_idx = sample[sample_idx].bridge_idx});
                    ++sample_idx;
                }
            }

            while (list_idx < cur_list.size()) {
                cur_aug.push_back(AugmentedElement{
                    .value = cur_list[list_idx], .orig_idx = list_idx, .bridge_idx = 0});
                ++list_idx;
            }

            while (sample_idx < sample.size()) {
                cur_aug.push_back(AugmentedElement{.value = sample[sample_idx].value,
                                                   .orig_idx = cur_list.size(),
                                                   .bridge_idx = sample[sample_idx].bridge_idx});
                ++sample_idx;
            }

            cur_aug.push_back(
                AugmentedElement{.value = make_sentinel_val(materialized_lists, cur_idx),
                                 .orig_idx = cur_list.size(),
                                 .bridge_idx = next_real_count});

            // Compute exact lower_bound orig_idx and bridge_idx via synchronized linear scans
            const std::size_t cur_real_count = cur_aug.size() - 1;

            // 1. Compute orig_idx
            compute_bound_indices(
                cur_aug, cur_real_count, cur_list, cur_list.size(), comp_,
                [](AugmentedElement& elem, std::size_t idx) noexcept { elem.orig_idx = idx; });

            // 2. Compute bridge_idx
            compute_bound_indices(
                cur_aug, cur_real_count, next_aug, next_real_count, comp_,
                [](AugmentedElement& elem, std::size_t idx) noexcept { elem.bridge_idx = idx; },
                &AugmentedElement::value);
        }

        // Cache direct pointers and sizes for branchless downward walk
        for (std::size_t i = 0; i < k_; ++i) {
            aug_data_[i] = augmented_lists_[i].data();
            aug_sizes_[i] = augmented_lists_[i].size();
        }
    }

    /**
     * @brief Returns the lower_bound index of target across all K lists in O(log N + K) time.
     *
     * @param target The search key.
     * @return std::vector<std::size_t> Vector of size K containing lower_bound indices.
     */
    [[nodiscard]] std::vector<std::size_t> search_all(const T& target) const {
        std::vector<std::size_t> result(k_);
        search_all_to_span(target, result);
        return result;
    }

    /**
     * @brief Writes lower_bound indices directly into a user-provided output span (zero
     * allocation).
     *
     * @param target The search key.
     * @param out_indices Output span of size at least list_count().
     */
    void search_all_to_span(const T& target, std::span<std::size_t> out_indices) const noexcept {
        if (k_ == 0) {
            return;
        }

        assert(out_indices.size() >= k_);

        // Top level binary search on M_0
        const std::size_t size0 = aug_sizes_[0];
        if (size0 == 0) {
            for (std::size_t i = 0; i < k_; ++i) {
                out_indices[i] = 0;
            }
            return;
        }

        const auto* aug0 = aug_data_[0];
        const std::size_t real0_count = size0 - 1; // Exclude sentinel

        const auto* it = branchless_lower_bound(aug0, aug0 + real0_count, target, comp_,
                                                &AugmentedElement::value);
        std::size_t pos = static_cast<std::size_t>(it - aug0);

        out_indices[0] = aug0[pos].orig_idx;

        // Constant-time branchless downward walk through levels 1 .. K-1
        for (std::size_t i = 1; i < k_; ++i) {
            const auto* prev_aug = aug_data_[i - 1];
            const auto* cur_aug = aug_data_[i];

            const std::size_t bridge_idx = prev_aug[pos].bridge_idx;
            const std::size_t check_idx = bridge_idx - (bridge_idx != 0);
            const bool match = (bridge_idx != 0) & !comp_(cur_aug[check_idx].value, target);
            pos = bridge_idx - (match ? 1 : 0);

            out_indices[i] = cur_aug[pos].orig_idx;
        }
    }

    /**
     * @brief Returns the number of lists K in the cascade.
     */
    [[nodiscard]] std::size_t list_count() const noexcept {
        return k_;
    }

    /**
     * @brief Returns the total number of augmented elements stored across all K levels.
     */
    [[nodiscard]] std::size_t total_augmented_elements() const noexcept {
        std::size_t total = 0;
        for (const auto& list : augmented_lists_) {
            total += list.size();
        }
        return total;
    }

private:
    struct AugmentedElement {
        T value;
        std::size_t orig_idx{0};   // Index in original list L_i
        std::size_t bridge_idx{0}; // Pointer/index into augmented list M_{i+1}
    };

    template <typename RefContainer, typename Setter, typename Proj = std::identity>
    static void compute_bound_indices(std::vector<AugmentedElement>& target_aug,
                                      std::size_t target_count, const RefContainer& ref_list,
                                      std::size_t ref_count, const Compare& comp, Setter setter,
                                      Proj proj = {}) {
        std::size_t ref_ptr = 0;
        for (std::size_t j = 0; j < target_count; ++j) {
            while (ref_ptr < ref_count &&
                   comp(std::invoke(proj, ref_list[ref_ptr]), target_aug[j].value)) {
                ++ref_ptr;
            }
            setter(target_aug[j], ref_ptr);
        }
    }

    static T make_sentinel_val(const std::vector<std::vector<T>>& lists, std::size_t cur_idx) {
        if constexpr (std::default_initializable<T>) {
            return T{};
        } else {
            if (!lists[cur_idx].empty()) {
                return lists[cur_idx].back();
            }
            for (const auto& l : lists) {
                if (!l.empty()) {
                    return l.back();
                }
            }
            throw std::invalid_argument("FractionalCascading requires at least one element for "
                                        "non-default-constructible T");
        }
    }

    std::size_t k_{0};
    Compare comp_{};
    std::vector<std::vector<AugmentedElement>> augmented_lists_;
    std::vector<const AugmentedElement*> aug_data_;
    std::vector<std::size_t> aug_sizes_;
};

} // namespace algoat::searching

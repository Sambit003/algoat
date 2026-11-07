/**
 * @file veb_tree.hpp
 * @brief Van Emde Boas (vEB) Tree integer predecessor engine.
 *
 * Implements Peter van Emde Boas's O(log log U) priority queue and predecessor search structure
 * over a bounded integer universe [0, 2^Bits - 1], with lazy cluster allocation to prevent
 * Theta(U) memory blowup.
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace algoat::searching {

// Forward declaration
template <std::size_t Bits = 32> class VanEmdeBoasTree;

/**
 * @brief Base-case specialization of VanEmdeBoasTree for 1-bit universe (U = 2, keys in {0, 1}).
 *
 * Eliminates all dynamic allocations, recursive clusters, and summary structures,
 * executing all operations in O(1) branchless bitwise operations.
 */
template <> class VanEmdeBoasTree<1> {
public:
    using key_type = uint32_t;
    static constexpr key_type universe_size = 2;

    constexpr VanEmdeBoasTree() noexcept = default;
    ~VanEmdeBoasTree() = default;

    constexpr VanEmdeBoasTree(const VanEmdeBoasTree&) noexcept = default;
    constexpr VanEmdeBoasTree& operator=(const VanEmdeBoasTree&) noexcept = default;
    VanEmdeBoasTree(VanEmdeBoasTree&&) noexcept = default;
    VanEmdeBoasTree& operator=(VanEmdeBoasTree&&) noexcept = default;

    [[nodiscard]] constexpr bool empty() const noexcept {
        return mask_ == 0;
    }

    [[nodiscard]] constexpr std::optional<key_type> min() const noexcept {
        if (mask_ & 1U)
            return 0;
        if (mask_ & 2U)
            return 1;
        return std::nullopt;
    }

    [[nodiscard]] constexpr std::optional<key_type> max() const noexcept {
        if (mask_ & 2U)
            return 1;
        if (mask_ & 1U)
            return 0;
        return std::nullopt;
    }

    [[nodiscard]] constexpr bool contains(key_type x) const noexcept {
        return x < 2 && ((mask_ & (1U << x)) != 0);
    }

    [[nodiscard]] constexpr bool find(key_type x) const noexcept {
        return contains(x);
    }

    [[nodiscard]] constexpr std::optional<key_type> successor(key_type x) const noexcept {
        if (x == 0 && (mask_ & 2U)) {
            return 1;
        }
        return std::nullopt;
    }

    [[nodiscard]] constexpr std::optional<key_type> predecessor(key_type x) const noexcept {
        if (x > 1) {
            if (mask_ & 2U)
                return 1;
            if (mask_ & 1U)
                return 0;
        } else if (x == 1) {
            if (mask_ & 1U)
                return 0;
        }
        return std::nullopt;
    }

    constexpr void insert(key_type x) noexcept {
        if (x < 2) {
            mask_ |= static_cast<uint8_t>(1U << x);
        }
    }

    constexpr void erase(key_type x) noexcept {
        if (x < 2) {
            mask_ &= static_cast<uint8_t>(~(1U << x));
        }
    }

    constexpr void clear() noexcept {
        mask_ = 0;
    }

private:
    uint8_t mask_{0};
};

/**
 * @brief Van Emde Boas (vEB) Tree associative integer priority queue and predecessor search
 * structure.
 *
 * Executes contains, successor, predecessor, insert, and erase in strictly O(log log U) time,
 * completely independent of the number of elements N stored in the tree.
 *
 * @tparam Bits Number of bits representing the universe [0, 2^Bits - 1], in range [1, 64].
 */
template <std::size_t Bits> class VanEmdeBoasTree {
public:
    static_assert(Bits > 0 && Bits <= 64, "Universe bit width must be in [1, 64]");
    using key_type = std::conditional_t<(Bits <= 32), uint32_t, uint64_t>;
    static constexpr std::size_t upper_bits = (Bits + 1) / 2;
    static constexpr std::size_t lower_bits = Bits / 2;

    static constexpr key_type universe_size =
        (Bits == 32 || Bits == 64) ? ~key_type(0) : (key_type(1) << (Bits < 64 ? Bits : 0));

    VanEmdeBoasTree() = default;
    ~VanEmdeBoasTree() = default;

    VanEmdeBoasTree(const VanEmdeBoasTree& other) : min_(other.min_), max_(other.max_) {
        if (other.summary_) {
            summary_ = std::make_unique<VanEmdeBoasTree<upper_bits>>(*other.summary_);
        }
        if constexpr (upper_bits > 16) {
            for (const auto& [cluster_id, cluster_node] : other.cluster_) {
                if (cluster_node) {
                    cluster_[cluster_id] =
                        std::make_unique<VanEmdeBoasTree<lower_bits>>(*cluster_node);
                }
            }
        } else {
            cluster_.resize(other.cluster_.size());
            for (std::size_t i = 0; i < other.cluster_.size(); ++i) {
                if (other.cluster_[i]) {
                    cluster_[i] = std::make_unique<VanEmdeBoasTree<lower_bits>>(*other.cluster_[i]);
                }
            }
        }
    }

    VanEmdeBoasTree& operator=(const VanEmdeBoasTree& other) {
        if (this != &other) {
            VanEmdeBoasTree tmp(other);
            *this = std::move(tmp);
        }
        return *this;
    }

    VanEmdeBoasTree(VanEmdeBoasTree&&) noexcept = default;
    VanEmdeBoasTree& operator=(VanEmdeBoasTree&&) noexcept = default;

    [[nodiscard]] bool empty() const noexcept {
        return !min_.has_value();
    }

    [[nodiscard]] std::optional<key_type> min() const noexcept {
        return min_;
    }

    [[nodiscard]] std::optional<key_type> max() const noexcept {
        return max_;
    }

    [[nodiscard]] bool contains(key_type x) const noexcept {
        if (empty())
            return false;
        if (x == *min_ || x == *max_)
            return true;
        if (x < *min_ || x > *max_)
            return false;

        const auto* cluster_ptr = get_cluster(high(x));
        return cluster_ptr ? cluster_ptr->contains(low(x)) : false;
    }

    [[nodiscard]] bool find(key_type x) const noexcept {
        return contains(x);
    }

    [[nodiscard]] std::optional<key_type> successor(key_type x) const noexcept {
        if (empty() || x >= *max_)
            return std::nullopt;
        if (x < *min_)
            return min_;

        const auto cluster_idx = high(x);
        const auto cluster_offset = low(x);

        // 1. If cluster cluster_idx has elements and cluster_offset < cluster->max()
        if (const auto* cluster_ptr = get_cluster(cluster_idx);
            cluster_ptr && !cluster_ptr->empty() && cluster_offset < *cluster_ptr->max()) {
            auto offset = cluster_ptr->successor(cluster_offset);
            if (offset)
                return index(cluster_idx, *offset);
        }

        // 2. Otherwise query summary for next non-empty cluster
        if (!summary_ || summary_->empty())
            return std::nullopt;
        auto next_cluster_idx = summary_->successor(cluster_idx);
        if (!next_cluster_idx)
            return std::nullopt;

        const auto* next_cluster_ptr = get_cluster(*next_cluster_idx);
        if (!next_cluster_ptr || next_cluster_ptr->empty())
            return std::nullopt;
        return index(*next_cluster_idx, *next_cluster_ptr->min());
    }

    [[nodiscard]] std::optional<key_type> predecessor(key_type x) const noexcept {
        if (empty() || x <= *min_)
            return std::nullopt;
        if (x > *max_)
            return max_;

        const auto cluster_idx = high(x);
        const auto cluster_offset = low(x);

        // 1. If cluster cluster_idx has elements and cluster_offset > cluster->min()
        if (const auto* cluster_ptr = get_cluster(cluster_idx);
            cluster_ptr && !cluster_ptr->empty() && cluster_offset > *cluster_ptr->min()) {
            auto offset = cluster_ptr->predecessor(cluster_offset);
            if (offset)
                return index(cluster_idx, *offset);
        }

        // 2. Otherwise query summary for preceding cluster
        std::optional<key_type> pred_cluster_idx;
        if (summary_ && !summary_->empty()) {
            pred_cluster_idx = summary_->predecessor(cluster_idx);
        }

        if (pred_cluster_idx) {
            const auto* prev_cluster_ptr = get_cluster(*pred_cluster_idx);
            if (prev_cluster_ptr && !prev_cluster_ptr->empty()) {
                return index(*pred_cluster_idx, *prev_cluster_ptr->max());
            }
        }

        // 3. If no lower cluster exists, the predecessor might be min_
        if (min_ && x > *min_)
            return min_;
        return std::nullopt;
    }

    void insert(key_type x) {
        if constexpr (Bits != 32 && Bits != 64) {
            if (x >= universe_size)
                return;
        }

        if (empty()) {
            min_ = x;
            max_ = x;
            return;
        }

        if (x == *min_ || x == *max_)
            return;

        // Invariant 1: min_ is not stored in clusters
        if (x < *min_) {
            std::swap(x, *min_);
        }

        if (x > *max_) {
            max_ = x;
        }

        const auto cluster_idx = high(x);
        const auto cluster_offset = low(x);

        auto* cluster_ptr = ensure_cluster(cluster_idx);

        if (cluster_ptr->empty()) {
            if (!summary_) {
                summary_ = std::make_unique<VanEmdeBoasTree<upper_bits>>();
            }
            summary_->insert(cluster_idx);
        }

        cluster_ptr->insert(cluster_offset);
    }

    void erase(key_type x) noexcept {
        if (empty())
            return;

        // Case 1: Tree contains exactly one element
        if (*min_ == *max_) {
            if (x == *min_) {
                min_.reset();
                max_.reset();
            }
            return;
        }

        // Case 2: Deleting min_
        if (x == *min_) {
            if (!summary_ || summary_->empty()) {
                min_ = max_;
                return;
            }
            const auto first_cluster_idx = *summary_->min();
            const auto* cluster_ptr = get_cluster(first_cluster_idx);
            if (!cluster_ptr || cluster_ptr->empty()) {
                min_ = max_;
                return;
            }
            const auto new_min_offset = *cluster_ptr->min();
            x = index(first_cluster_idx, new_min_offset);
            min_ = x; // Overwrite min_ with next smallest element
        }

        const auto cluster_idx = high(x);
        const auto cluster_offset = low(x);

        if (auto* cluster_ptr = get_cluster(cluster_idx)) {
            if (!cluster_ptr->contains(cluster_offset))
                return;

            cluster_ptr->erase(cluster_offset);

            if (cluster_ptr->empty()) {
                erase_cluster(cluster_idx); // Reclaim empty cluster memory
                summary_->erase(cluster_idx);

                if (x == *max_) {
                    if (summary_->empty()) {
                        max_ = min_;
                    } else {
                        const auto max_cluster_idx = *summary_->max();
                        const auto* max_cluster_ptr = get_cluster(max_cluster_idx);
                        if (max_cluster_ptr && !max_cluster_ptr->empty()) {
                            max_ = index(max_cluster_idx, *max_cluster_ptr->max());
                        } else {
                            max_ = min_;
                        }
                    }
                }
            } else if (x == *max_) {
                max_ = index(cluster_idx, *cluster_ptr->max());
            }
        }
    }

    void clear() noexcept {
        min_.reset();
        max_.reset();
        summary_.reset();
        clear_clusters();
    }

private:
    static constexpr key_type low_mask =
        (lower_bits >= 64) ? ~key_type(0) : ((key_type(1) << lower_bits) - key_type(1));

    static constexpr key_type high(key_type x) noexcept {
        if constexpr (lower_bits >= 64)
            return 0;
        else
            return x >> lower_bits;
    }

    static constexpr key_type low(key_type x) noexcept {
        return x & low_mask;
    }

    static constexpr key_type index(key_type h, key_type l) noexcept {
        if constexpr (lower_bits >= 64)
            return l;
        else
            return (h << lower_bits) | l;
    }

    // Cluster storage: vector for upper_bits <= 16 (Bits <= 32), map for upper_bits > 16 (Bits =
    // 64)
    using cluster_container = std::conditional_t<
        (upper_bits > 16),
        std::unordered_map<key_type, std::unique_ptr<VanEmdeBoasTree<lower_bits>>>,
        std::vector<std::unique_ptr<VanEmdeBoasTree<lower_bits>>>>;

    [[nodiscard]] VanEmdeBoasTree<lower_bits>* get_cluster(key_type h) const noexcept {
        if constexpr (upper_bits > 16) {
            auto it = cluster_.find(h);
            return (it != cluster_.end()) ? it->second.get() : nullptr;
        } else {
            return (h < cluster_.size()) ? cluster_[h].get() : nullptr;
        }
    }

    VanEmdeBoasTree<lower_bits>* ensure_cluster(key_type h) {
        if constexpr (upper_bits > 16) {
            auto& ptr = cluster_[h];
            if (!ptr) {
                ptr = std::make_unique<VanEmdeBoasTree<lower_bits>>();
            }
            return ptr.get();
        } else {
            if (h >= cluster_.size()) {
                cluster_.resize(static_cast<std::size_t>(h + 1));
            }
            if (!cluster_[h]) {
                cluster_[h] = std::make_unique<VanEmdeBoasTree<lower_bits>>();
            }
            return cluster_[h].get();
        }
    }

    void erase_cluster(key_type h) noexcept {
        if constexpr (upper_bits > 16) {
            cluster_.erase(h);
        } else {
            if (h < cluster_.size()) {
                cluster_[h].reset();
            }
        }
    }

    void clear_clusters() noexcept {
        cluster_.clear();
    }

    std::optional<key_type> min_{};
    std::optional<key_type> max_{};

    // Summary vEB tree tracking non-empty clusters
    std::unique_ptr<VanEmdeBoasTree<upper_bits>> summary_{};
    // Lazy array or hash map of active child clusters
    cluster_container cluster_{};
};

// Explicit type aliases for common universe sizes
using VEBTree16 = VanEmdeBoasTree<16>;
using VEBTree32 = VanEmdeBoasTree<32>;
using VEBTree64 = VanEmdeBoasTree<64>;

} // namespace algoat::searching

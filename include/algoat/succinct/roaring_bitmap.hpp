#pragma once

#include "algoat/succinct/detail/containers.hpp"
#include "algoat/succinct/detail/iterator.hpp"
#include "algoat/succinct/detail/operations.hpp"
#include "algoat/succinct/detail/serialization.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

namespace algoat::succinct {

class RoaringBitmap {
public:
    using const_iterator = detail::RoaringBitmapIterator;

    RoaringBitmap() = default;

    RoaringBitmap(std::initializer_list<uint32_t> init) {
        for (uint32_t val : init) {
            add(val);
        }
    }

    explicit RoaringBitmap(std::span<const uint32_t> sorted_vals) {
        for (uint32_t val : sorted_vals) {
            add(val);
        }
    }

    RoaringBitmap(const RoaringBitmap& other) : keys_(other.keys_) {
        containers_.reserve(other.containers_.size());
        for (const auto& c : other.containers_) {
            containers_.push_back(c ? c->clone() : nullptr);
        }
    }

    RoaringBitmap(RoaringBitmap&& other) noexcept = default;

    RoaringBitmap& operator=(const RoaringBitmap& other) {
        if (this != &other) {
            keys_ = other.keys_;
            containers_.clear();
            containers_.reserve(other.containers_.size());
            for (const auto& c : other.containers_) {
                containers_.push_back(c ? c->clone() : nullptr);
            }
        }
        return *this;
    }

    RoaringBitmap& operator=(RoaringBitmap&& other) noexcept = default;

    ~RoaringBitmap() = default;

    /// \brief Inserts an integer value. Returns true if newly added.
    bool add(uint32_t val) {
        const uint16_t key = static_cast<uint16_t>(val >> 16);
        const uint16_t low = static_cast<uint16_t>(val & 0xFFFF);
        const int32_t idx = find_key_index(key);

        if (idx >= 0) {
            const size_t pos = static_cast<size_t>(idx);
            bool added = containers_[pos]->add(low);
            if (added) {
                if (containers_[pos]->type() == detail::ContainerType::Array) {
                    if (containers_[pos]->cardinality() > detail::ARRAY_MAX_CAPACITY) {
                        const auto* ac =
                            static_cast<const detail::ArrayContainer*>(containers_[pos].get());
                        containers_[pos] = detail::array_to_bitset(*ac);
                    }
                } else if (containers_[pos]->type() == detail::ContainerType::Run) {
                    const auto* rc =
                        static_cast<const detail::RunContainer*>(containers_[pos].get());
                    if (2 + 4 * rc->runs.size() > detail::BITSET_BYTES) {
                        containers_[pos] = detail::run_to_bitset(*rc);
                    }
                }
            }
            return added;
        }

        const size_t insert_pos = static_cast<size_t>(-idx - 1);
        auto ac = std::make_shared<detail::ArrayContainer>();
        ac->values.push_back(low);
        keys_.insert(keys_.begin() + static_cast<std::ptrdiff_t>(insert_pos), key);
        containers_.insert(containers_.begin() + static_cast<std::ptrdiff_t>(insert_pos), ac);
        return true;
    }

    /// \brief Removes an integer value. Returns true if removed.
    bool remove(uint32_t val) {
        const uint16_t key = static_cast<uint16_t>(val >> 16);
        const uint16_t low = static_cast<uint16_t>(val & 0xFFFF);
        const int32_t idx = find_key_index(key);
        if (idx < 0)
            return false;

        const size_t pos = static_cast<size_t>(idx);
        bool removed = containers_[pos]->remove(low);
        if (removed) {
            if (containers_[pos]->cardinality() == 0) {
                keys_.erase(keys_.begin() + static_cast<std::ptrdiff_t>(pos));
                containers_.erase(containers_.begin() + static_cast<std::ptrdiff_t>(pos));
            } else if (containers_[pos]->type() == detail::ContainerType::Bitset &&
                       containers_[pos]->cardinality() <= detail::ARRAY_MAX_CAPACITY) {
                const auto* bc =
                    static_cast<const detail::BitsetContainer*>(containers_[pos].get());
                containers_[pos] = detail::bitset_to_array(*bc);
            } else if (containers_[pos]->type() == detail::ContainerType::Run &&
                       containers_[pos]->cardinality() <= detail::ARRAY_MAX_CAPACITY) {
                const auto* rc = static_cast<const detail::RunContainer*>(containers_[pos].get());
                const uint32_t card = rc->cardinality();
                if (2 * card < 2 + 4 * rc->runs.size()) {
                    containers_[pos] = detail::run_to_array(*rc);
                }
            }
        }
        return removed;
    }

    /// \brief Tests membership in O(1) or O(log K) time.
    [[nodiscard]] bool contains(uint32_t val) const noexcept {
        const uint16_t key = static_cast<uint16_t>(val >> 16);
        const uint16_t low = static_cast<uint16_t>(val & 0xFFFF);
        const int32_t idx = find_key_index(key);
        if (idx >= 0) {
            return containers_[static_cast<size_t>(idx)]->contains(low);
        }
        return false;
    }

    /// \brief Computes total cardinality.
    [[nodiscard]] uint64_t cardinality() const noexcept {
        uint64_t total = 0;
        for (const auto& c : containers_) {
            if (c)
                total += c->cardinality();
        }
        return total;
    }

    /// \brief Returns true if bitmap contains zero elements.
    [[nodiscard]] bool empty() const noexcept {
        return containers_.empty();
    }

    /// \brief Converts Bitset/Array containers to Run containers if compression improves.
    void run_optimize() noexcept {
        for (size_t i = 0; i < containers_.size(); ++i) {
            if (!containers_[i])
                continue;
            if (containers_[i]->type() == detail::ContainerType::Array) {
                const auto* ac = static_cast<const detail::ArrayContainer*>(containers_[i].get());
                auto rc = detail::array_to_run(*ac);
                if (2 + 4 * rc->runs.size() < 2 * ac->values.size()) {
                    containers_[i] = rc;
                }
            } else if (containers_[i]->type() == detail::ContainerType::Bitset) {
                const auto* bc = static_cast<const detail::BitsetContainer*>(containers_[i].get());
                uint32_t r = detail::count_runs_in_bitset(*bc);
                if (r <= detail::RUN_MAX_CAPACITY_IN_BITSET) {
                    containers_[i] = detail::bitset_to_run(*bc);
                }
            }
        }
    }

    // Fast In-Place and Out-of-Place Bitwise Set Operations
    RoaringBitmap& operator&=(const RoaringBitmap& other) {
        std::vector<uint16_t> new_keys;
        std::vector<std::shared_ptr<Container>> new_containers;

        size_t i = 0, j = 0;
        while (i < keys_.size() && j < other.keys_.size()) {
            if (keys_[i] == other.keys_[j]) {
                auto res = detail::intersect_containers(*containers_[i], *other.containers_[j]);
                if (res && res->cardinality() > 0) {
                    new_keys.push_back(keys_[i]);
                    new_containers.push_back(res);
                }
                ++i;
                ++j;
            } else if (keys_[i] < other.keys_[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        keys_ = std::move(new_keys);
        containers_ = std::move(new_containers);
        return *this;
    }

    RoaringBitmap& operator|=(const RoaringBitmap& other) {
        std::vector<uint16_t> new_keys;
        std::vector<std::shared_ptr<Container>> new_containers;

        size_t i = 0, j = 0;
        while (i < keys_.size() && j < other.keys_.size()) {
            if (keys_[i] == other.keys_[j]) {
                auto res = detail::union_containers(*containers_[i], *other.containers_[j]);
                new_keys.push_back(keys_[i]);
                new_containers.push_back(res);
                ++i;
                ++j;
            } else if (keys_[i] < other.keys_[j]) {
                new_keys.push_back(keys_[i]);
                new_containers.push_back(containers_[i]->clone());
                ++i;
            } else {
                new_keys.push_back(other.keys_[j]);
                new_containers.push_back(other.containers_[j]->clone());
                ++j;
            }
        }
        detail::append_cloned_tail(new_keys, new_containers, keys_, containers_, i);
        detail::append_cloned_tail(new_keys, new_containers, other.keys_, other.containers_, j);
        keys_ = std::move(new_keys);
        containers_ = std::move(new_containers);
        return *this;
    }

    RoaringBitmap& operator^=(const RoaringBitmap& other) {
        std::vector<uint16_t> new_keys;
        std::vector<std::shared_ptr<Container>> new_containers;

        size_t i = 0, j = 0;
        while (i < keys_.size() && j < other.keys_.size()) {
            if (keys_[i] == other.keys_[j]) {
                auto res = detail::xor_containers(*containers_[i], *other.containers_[j]);
                if (res && res->cardinality() > 0) {
                    new_keys.push_back(keys_[i]);
                    new_containers.push_back(res);
                }
                ++i;
                ++j;
            } else if (keys_[i] < other.keys_[j]) {
                new_keys.push_back(keys_[i]);
                new_containers.push_back(containers_[i]->clone());
                ++i;
            } else {
                new_keys.push_back(other.keys_[j]);
                new_containers.push_back(other.containers_[j]->clone());
                ++j;
            }
        }
        detail::append_cloned_tail(new_keys, new_containers, keys_, containers_, i);
        detail::append_cloned_tail(new_keys, new_containers, other.keys_, other.containers_, j);
        keys_ = std::move(new_keys);
        containers_ = std::move(new_containers);
        return *this;
    }

    RoaringBitmap& operator-=(const RoaringBitmap& other) {
        std::vector<uint16_t> new_keys;
        std::vector<std::shared_ptr<Container>> new_containers;

        size_t i = 0, j = 0;
        while (i < keys_.size() && j < other.keys_.size()) {
            if (keys_[i] == other.keys_[j]) {
                auto res = detail::difference_containers(*containers_[i], *other.containers_[j]);
                if (res && res->cardinality() > 0) {
                    new_keys.push_back(keys_[i]);
                    new_containers.push_back(res);
                }
                ++i;
                ++j;
            } else if (keys_[i] < other.keys_[j]) {
                new_keys.push_back(keys_[i]);
                new_containers.push_back(containers_[i]->clone());
                ++i;
            } else {
                ++j;
            }
        }
        detail::append_cloned_tail(new_keys, new_containers, keys_, containers_, i);
        keys_ = std::move(new_keys);
        containers_ = std::move(new_containers);
        return *this;
    }

    friend RoaringBitmap operator&(const RoaringBitmap& a, const RoaringBitmap& b) {
        RoaringBitmap res = a;
        res &= b;
        return res;
    }

    friend RoaringBitmap operator|(const RoaringBitmap& a, const RoaringBitmap& b) {
        RoaringBitmap res = a;
        res |= b;
        return res;
    }

    friend RoaringBitmap operator^(const RoaringBitmap& a, const RoaringBitmap& b) {
        RoaringBitmap res = a;
        res ^= b;
        return res;
    }

    friend RoaringBitmap operator-(const RoaringBitmap& a, const RoaringBitmap& b) {
        RoaringBitmap res = a;
        res -= b;
        return res;
    }

    /// \brief Computes Jaccard index |A & B| / |A | B| directly without full allocation.
    [[nodiscard]] static double jaccard_index(const RoaringBitmap& a,
                                              const RoaringBitmap& b) noexcept {
        const uint64_t ca = a.cardinality();
        const uint64_t cb = b.cardinality();
        if (ca == 0 && cb == 0)
            return 1.0;

        uint64_t inter = 0;
        size_t i = 0, j = 0;
        while (i < a.keys_.size() && j < b.keys_.size()) {
            if (a.keys_[i] == b.keys_[j]) {
                inter += detail::and_cardinality(*a.containers_[i], *b.containers_[j]);
                ++i;
                ++j;
            } else if (a.keys_[i] < b.keys_[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        const uint64_t union_card = ca + cb - inter;
        if (union_card == 0)
            return 1.0;
        return static_cast<double>(inter) / static_cast<double>(union_card);
    }

    /// \brief Serialization conforming to official standard Roaring format.
    [[nodiscard]] std::vector<uint8_t> serialize() const {
        return detail::Serializer::serialize(keys_, containers_);
    }

    static RoaringBitmap deserialize(std::span<const uint8_t> bytes) {
        RoaringBitmap bm;
        detail::Serializer::deserialize(bytes, bm.keys_, bm.containers_);
        return bm;
    }

    /// \brief C++20 Forward Iterator over stored integers.
    [[nodiscard]] const_iterator begin() const noexcept {
        return const_iterator(&keys_, &containers_, 0);
    }

    [[nodiscard]] const_iterator end() const noexcept {
        return const_iterator(&keys_, &containers_, containers_.size());
    }

    [[nodiscard]] size_t memory_usage_bytes() const noexcept {
        size_t total = sizeof(*this);
        total += keys_.capacity() * sizeof(uint16_t);
        total += containers_.capacity() * sizeof(std::shared_ptr<Container>);
        for (const auto& c : containers_) {
            if (c)
                total += c->memory_usage_bytes();
        }
        return total;
    }

private:
    using Container = detail::Container;
    using ArrayContainer = detail::ArrayContainer;
    using BitsetContainer = detail::BitsetContainer;
    using RunContainer = detail::RunContainer;

    std::vector<uint16_t> keys_;
    std::vector<std::shared_ptr<Container>> containers_;

    [[nodiscard]] int32_t find_key_index(uint16_t key) const noexcept {
        auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
        if (it != keys_.end() && *it == key) {
            return static_cast<int32_t>(std::distance(keys_.begin(), it));
        }
        return -1 - static_cast<int32_t>(std::distance(keys_.begin(), it));
    }
};

static_assert(std::ranges::forward_range<RoaringBitmap>);

} // namespace algoat::succinct

#pragma once

#include "algoat/succinct/detail/containers.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

namespace algoat::succinct {

class RoaringBitmap;

namespace detail {

class RoaringBitmapIterator {
public:
    using iterator_concept = std::forward_iterator_tag;
    using iterator_category = std::forward_iterator_tag;
    using value_type = uint32_t;
    using difference_type = std::ptrdiff_t;
    using pointer = const uint32_t*;
    using reference = uint32_t;

    RoaringBitmapIterator() = default;

    RoaringBitmapIterator(const std::vector<uint16_t>* keys,
                          const std::vector<std::shared_ptr<Container>>* containers,
                          size_t container_idx)
        : keys_(keys), containers_(containers), container_idx_(container_idx) {
        if (keys_ && containers_ && container_idx_ < containers_->size()) {
            init_container();
        }
    }

    reference operator*() const noexcept {
        return current_val_;
    }

    RoaringBitmapIterator& operator++() noexcept {
        advance();
        return *this;
    }

    RoaringBitmapIterator operator++(int) noexcept {
        RoaringBitmapIterator tmp = *this;
        advance();
        return tmp;
    }

    bool operator==(const RoaringBitmapIterator& other) const noexcept {
        if (is_end() && other.is_end())
            return true;
        if (is_end() != other.is_end())
            return false;
        return (keys_ == other.keys_) && (containers_ == other.containers_) &&
               (container_idx_ == other.container_idx_) &&
               (in_container_idx_ == other.in_container_idx_) && (run_idx_ == other.run_idx_) &&
               (val_in_run_ == other.val_in_run_);
    }

    bool operator!=(const RoaringBitmapIterator& other) const noexcept {
        return !(*this == other);
    }

private:
    [[nodiscard]] bool is_end() const noexcept {
        return !containers_ || container_idx_ >= containers_->size();
    }

    void init_container() noexcept {
        while (container_idx_ < containers_->size()) {
            const auto& c = (*containers_)[container_idx_];
            if (c->cardinality() > 0) {
                const uint32_t key_prefix = static_cast<uint32_t>((*keys_)[container_idx_]) << 16;
                if (c->type() == ContainerType::Array) {
                    const auto* ac = static_cast<const ArrayContainer*>(c.get());
                    in_container_idx_ = 0;
                    current_val_ = key_prefix | ac->values[0];
                    return;
                } else if (c->type() == ContainerType::Bitset) {
                    const auto* bc = static_cast<const BitsetContainer*>(c.get());
                    in_container_idx_ = 0;
                    if (find_next_bitset_val(bc, 0)) {
                        return;
                    }
                } else if (c->type() == ContainerType::Run) {
                    const auto* rc = static_cast<const RunContainer*>(c.get());
                    if (!rc->runs.empty()) {
                        run_idx_ = 0;
                        val_in_run_ = 0;
                        current_val_ = key_prefix | rc->runs[0].start;
                        return;
                    }
                }
            }
            ++container_idx_;
        }
    }

    bool find_next_bitset_val(const BitsetContainer* bc, uint32_t start_bit) noexcept {
        const uint32_t key_prefix = static_cast<uint32_t>((*keys_)[container_idx_]) << 16;
        auto [found, next_bit] = bc->next_set_bit(start_bit);
        if (found) {
            in_container_idx_ = next_bit;
            current_val_ = key_prefix | next_bit;
            return true;
        }
        return false;
    }

    void advance() noexcept {
        if (is_end())
            return;

        const auto& c = (*containers_)[container_idx_];
        const uint32_t key_prefix = static_cast<uint32_t>((*keys_)[container_idx_]) << 16;

        if (c->type() == ContainerType::Array) {
            const auto* ac = static_cast<const ArrayContainer*>(c.get());
            ++in_container_idx_;
            if (in_container_idx_ < ac->values.size()) {
                current_val_ = key_prefix | ac->values[in_container_idx_];
                return;
            }
        } else if (c->type() == ContainerType::Bitset) {
            const auto* bc = static_cast<const BitsetContainer*>(c.get());
            if (find_next_bitset_val(bc, static_cast<uint32_t>(in_container_idx_ + 1))) {
                return;
            }
        } else if (c->type() == ContainerType::Run) {
            const auto* rc = static_cast<const RunContainer*>(c.get());
            if (run_idx_ < rc->runs.size()) {
                if (val_in_run_ < rc->runs[run_idx_].length) {
                    ++val_in_run_;
                    current_val_ =
                        key_prefix | static_cast<uint32_t>(rc->runs[run_idx_].start + val_in_run_);
                    return;
                }
                ++run_idx_;
                val_in_run_ = 0;
                if (run_idx_ < rc->runs.size()) {
                    current_val_ = key_prefix | rc->runs[run_idx_].start;
                    return;
                }
            }
        }

        // Advance to next container
        ++container_idx_;
        in_container_idx_ = 0;
        run_idx_ = 0;
        val_in_run_ = 0;
        init_container();
    }

    const std::vector<uint16_t>* keys_{nullptr};
    const std::vector<std::shared_ptr<Container>>* containers_{nullptr};
    size_t container_idx_{0};
    size_t in_container_idx_{0};
    size_t run_idx_{0};
    uint32_t val_in_run_{0};
    uint32_t current_val_{0};
};

} // namespace detail
} // namespace algoat::succinct

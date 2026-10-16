/******************************************************************************
 * algoat/sorting/detail/ips4o/buffers.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "ips4o_fwd.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace algoat::sorting::detail::ips4o {
namespace detail {

/**
 * A single buffer block.
 */
template <class Cfg> class Sorter<Cfg>::Block {
    using iterator = typename Cfg::iterator;
    using diff_t = typename Cfg::difference_type;
    using value_type = typename Cfg::value_type;

public:
    static constexpr const bool kInitializedStorage =
        std::is_trivially_default_constructible_v<value_type>;
    static constexpr const bool kDestruct =
        !kInitializedStorage && !std::is_trivially_destructible_v<value_type>;

    value_type* data() {
        return static_cast<value_type*>(static_cast<void*>(storage_));
    }

    const value_type* data() const {
        return static_cast<const value_type*>(static_cast<const void*>(storage_));
    }

    const value_type& head() const {
        return *data();
    }

    void readFrom(iterator src) {
        if constexpr (kInitializedStorage) {
            std::move(src, src + Cfg::kBlockSize, data());
        } else {
            for (auto p = data(), end = p + Cfg::kBlockSize; p < end; ++p) {
                IPS4O_ASSUME_NOT(p == nullptr);
                new (static_cast<void*>(p)) value_type(std::move(*src++));
            }
        }
    }

    void readFrom(iterator src, const diff_t n) {
        if constexpr (kInitializedStorage) {
            std::move(src, src + n, data());
        } else {
            for (auto p = data(), end = p + n; p < end; ++p) {
                IPS4O_ASSUME_NOT(p == nullptr);
                new (static_cast<void*>(p)) value_type(std::move(*src++));
            }
        }
    }

    void reset(const diff_t n) {
        if constexpr (kDestruct) {
            for (auto p = data(), end = p + n; p < end; ++p)
                p->~value_type();
        }
    }

    void writeTo(Block& block) {
        if constexpr (kInitializedStorage) {
            std::move(data(), data() + Cfg::kBlockSize, block.data());
        } else {
            for (auto src = data(), dst = block.data(), end = src + Cfg::kBlockSize; src < end;
                 ++src, ++dst) {
                IPS4O_ASSUME_NOT(dst == nullptr);
                new (static_cast<void*>(dst)) value_type(std::move(*src));
            }
        }
        if constexpr (kDestruct) {
            for (auto p = data(), end = p + Cfg::kBlockSize; p < end; ++p)
                p->~value_type();
        }
    }

    void writeTo(iterator dest) {
        std::move(data(), data() + Cfg::kBlockSize, std::move(dest));
        if constexpr (kDestruct) {
            for (auto p = data(), end = p + Cfg::kBlockSize; p < end; ++p)
                p->~value_type();
        }
    }

private:
    struct alignas(alignof(value_type)) UninitStorage {
        std::byte data[sizeof(value_type)];
    };
    using storage_type = std::conditional_t<kInitializedStorage, value_type, UninitStorage>;
    storage_type storage_[Cfg::kBlockSize];
};

/**
 * Per-thread buffers for each bucket.
 */
template <class Cfg> class Sorter<Cfg>::Buffers {
    using diff_t = typename Cfg::difference_type;
    using value_type = typename Cfg::value_type;

public:
    explicit Buffers(char* storage) : storage_(static_cast<Block*>(static_cast<void*>(storage))) {
        for (diff_t i = 0; i < Cfg::kMaxBuckets; ++i) {
            resetBuffer(i);
            buffer_[i].end = buffer_[i].ptr + Cfg::kBlockSize;
        }
    }

    bool isFull(const int i) const {
        return buffer_[i].ptr == buffer_[i].end;
    }

    value_type* data(const int i) {
        return static_cast<value_type*>(static_cast<void*>(storage_)) + i * Cfg::kBlockSize;
    }

    diff_t size(const int i) const {
        return Cfg::kBlockSize - (buffer_[i].end - buffer_[i].ptr);
    }

    void reset(const int i) {
        if constexpr (Block::kDestruct) {
            for (auto p = data(i), end = p + size(i); p < end; ++p)
                p->~value_type();
        }
        resetBuffer(i);
    }

    void push(const int i, value_type&& value) {
        if constexpr (Block::kInitializedStorage) {
            *buffer_[i].ptr++ = std::move(value);
        } else {
            IPS4O_ASSUME_NOT(buffer_[i].ptr == nullptr);
            new (static_cast<void*>(buffer_[i].ptr++)) value_type(std::move(value));
        }
    }

    void writeTo(const int i, typename Cfg::iterator dest) {
        resetBuffer(i);
        auto ptr = buffer_[i].ptr;
        std::move(ptr, ptr + Cfg::kBlockSize, std::move(dest));

        if constexpr (Block::kDestruct) {
            for (const auto end = buffer_[i].end; ptr < end; ++ptr)
                ptr->~value_type();
        }
    }

private:
    struct Info {
        value_type* ptr;
        const value_type* end;
    };

    void resetBuffer(const int i) {
        buffer_[i].ptr =
            static_cast<value_type*>(static_cast<void*>(storage_)) + i * Cfg::kBlockSize;
    }

    Info buffer_[Cfg::kMaxBuckets];
    Block* storage_;

    static_assert(sizeof(Block) == sizeof(typename Cfg::value_type) * Cfg::kBlockSize,
                  "Block size mismatch.");
    static_assert(std::is_trivially_default_constructible_v<Block>,
                  "Block must be trivially default constructible.");
    static_assert(std::is_trivially_destructible_v<Block>, "Block must be trivially destructible.");
};

} // namespace detail
} // namespace algoat::sorting::detail::ips4o

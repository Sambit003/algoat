#pragma once

#include "algoat/succinct/detail/simd_ops.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace algoat::succinct::detail {

inline constexpr uint32_t ARRAY_MAX_CAPACITY = 4096;
inline constexpr uint32_t RUN_MAX_CAPACITY_IN_BITSET = 2047;

enum class ContainerType : uint8_t { Array, Bitset, Run };

struct Container {
    virtual ~Container() = default;

    [[nodiscard]] virtual ContainerType type() const noexcept = 0;
    [[nodiscard]] virtual uint32_t cardinality() const noexcept = 0;
    [[nodiscard]] virtual bool contains(uint16_t val) const noexcept = 0;
    virtual bool add(uint16_t val) = 0;
    virtual bool remove(uint16_t val) = 0;
    [[nodiscard]] virtual std::shared_ptr<Container> clone() const = 0;
    [[nodiscard]] virtual size_t memory_usage_bytes() const noexcept = 0;
};

struct ArrayContainer final : public Container {
    std::vector<uint16_t> values;

    ArrayContainer() = default;
    explicit ArrayContainer(std::vector<uint16_t> vals) : values(std::move(vals)) {}

    [[nodiscard]] ContainerType type() const noexcept override {
        return ContainerType::Array;
    }

    [[nodiscard]] uint32_t cardinality() const noexcept override {
        return static_cast<uint32_t>(values.size());
    }

    [[nodiscard]] bool contains(uint16_t val) const noexcept override {
        return std::binary_search(values.begin(), values.end(), val);
    }

    bool add(uint16_t val) override {
        auto it = std::lower_bound(values.begin(), values.end(), val);
        if (it != values.end() && *it == val) {
            return false;
        }
        values.insert(it, val);
        return true;
    }

    bool remove(uint16_t val) override {
        auto it = std::lower_bound(values.begin(), values.end(), val);
        if (it != values.end() && *it == val) {
            values.erase(it);
            return true;
        }
        return false;
    }

    [[nodiscard]] std::shared_ptr<Container> clone() const override {
        return std::make_shared<ArrayContainer>(*this);
    }

    [[nodiscard]] size_t memory_usage_bytes() const noexcept override {
        return sizeof(*this) + values.capacity() * sizeof(uint16_t);
    }
};

struct BitsetContainer final : public Container {
    std::array<uint64_t, BITSET_WORDS> words{};
    uint32_t card_{0};

    BitsetContainer() = default;

    explicit BitsetContainer(const ArrayContainer& arr) {
        words.fill(0);
        card_ = arr.cardinality();
        for (uint16_t val : arr.values) {
            words[val / 64] |= (1ULL << (val % 64));
        }
    }

    [[nodiscard]] ContainerType type() const noexcept override {
        return ContainerType::Bitset;
    }

    [[nodiscard]] uint32_t cardinality() const noexcept override {
        return card_;
    }

    [[nodiscard]] bool contains(uint16_t val) const noexcept override {
        return (words[val / 64] & (1ULL << (val % 64))) != 0;
    }

    bool add(uint16_t val) override {
        const uint64_t mask = 1ULL << (val % 64);
        const size_t idx = val / 64;
        if ((words[idx] & mask) == 0) {
            words[idx] |= mask;
            ++card_;
            return true;
        }
        return false;
    }

    bool remove(uint16_t val) override {
        const uint64_t mask = 1ULL << (val % 64);
        const size_t idx = val / 64;
        if ((words[idx] & mask) != 0) {
            words[idx] &= ~mask;
            --card_;
            return true;
        }
        return false;
    }

    void recompute_cardinality() noexcept {
        card_ = static_cast<uint32_t>(SimdOps::bitset_popcount(words.data()));
    }

    [[nodiscard]] std::pair<bool, uint32_t> next_set_bit(uint32_t start_bit) const noexcept {
        const size_t word_idx = start_bit / 64;
        const uint32_t bit_offset = start_bit % 64;
        if (word_idx < BITSET_WORDS) {
            const uint64_t w = words[word_idx] >> bit_offset;
            if (w != 0) {
                const auto next_bit = bit_offset + static_cast<uint32_t>(__builtin_ctzll(w));
                return {true, static_cast<uint32_t>(word_idx * 64 + next_bit)};
            }
            for (size_t wi = word_idx + 1; wi < BITSET_WORDS; ++wi) {
                if (words[wi] != 0) {
                    const auto next_bit = static_cast<uint32_t>(__builtin_ctzll(words[wi]));
                    return {true, static_cast<uint32_t>(wi * 64 + next_bit)};
                }
            }
        }
        return {false, 0};
    }

    [[nodiscard]] std::shared_ptr<Container> clone() const override {
        return std::make_shared<BitsetContainer>(*this);
    }

    [[nodiscard]] size_t memory_usage_bytes() const noexcept override {
        return sizeof(*this);
    }
};

struct Run {
    uint16_t start{0};
    uint16_t length{0}; // Length minus 1 (i.e. covers [start, start + length] inclusive)
};

struct RunContainer final : public Container {
    std::vector<Run> runs;

    RunContainer() = default;
    explicit RunContainer(std::vector<Run> r) : runs(std::move(r)) {}

    [[nodiscard]] ContainerType type() const noexcept override {
        return ContainerType::Run;
    }

    [[nodiscard]] uint32_t cardinality() const noexcept override {
        uint32_t total = 0;
        for (const auto& r : runs) {
            total += static_cast<uint32_t>(r.length) + 1;
        }
        return total;
    }

    [[nodiscard]] bool contains(uint16_t val) const noexcept override {
        if (runs.empty())
            return false;
        auto it = std::upper_bound(runs.begin(), runs.end(), val,
                                   [](uint16_t v, const Run& r) noexcept { return v < r.start; });
        if (it == runs.begin()) {
            return false;
        }
        --it;
        return (val >= it->start) && (val <= static_cast<uint32_t>(it->start) + it->length);
    }

    bool add(uint16_t val) override {
        if (contains(val))
            return false;

        // Find position to insert
        auto it = std::lower_bound(runs.begin(), runs.end(), val,
                                   [](const Run& r, uint16_t v) noexcept { return r.start < v; });

        // Try to merge with preceding or succeeding runs
        Run new_run{val, 0};
        runs.insert(it, new_run);
        normalize();
        return true;
    }

    bool remove(uint16_t val) override {
        if (runs.empty())
            return false;
        for (size_t i = 0; i < runs.size(); ++i) {
            uint32_t s = runs[i].start;
            uint32_t e = s + runs[i].length;
            if (val >= s && val <= e) {
                if (s == e) {
                    runs.erase(runs.begin() + static_cast<std::ptrdiff_t>(i));
                } else if (val == s) {
                    runs[i].start = static_cast<uint16_t>(s + 1);
                    runs[i].length = static_cast<uint16_t>(runs[i].length - 1);
                } else if (val == e) {
                    runs[i].length = static_cast<uint16_t>(runs[i].length - 1);
                } else {
                    // Split run
                    Run left{static_cast<uint16_t>(s), static_cast<uint16_t>(val - 1 - s)};
                    Run right{static_cast<uint16_t>(val + 1), static_cast<uint16_t>(e - (val + 1))};
                    runs[i] = left;
                    runs.insert(runs.begin() + static_cast<std::ptrdiff_t>(i + 1), right);
                }
                return true;
            }
        }
        return false;
    }

    void normalize() {
        if (runs.size() <= 1)
            return;
        std::vector<Run> merged;
        merged.reserve(runs.size());
        merged.push_back(runs[0]);

        for (size_t i = 1; i < runs.size(); ++i) {
            auto& last = merged.back();
            uint32_t last_end = static_cast<uint32_t>(last.start) + last.length;
            uint32_t cur_start = runs[i].start;
            uint32_t cur_end = cur_start + runs[i].length;

            if (cur_start <= last_end + 1) {
                uint32_t new_end = std::max(last_end, cur_end);
                last.length = static_cast<uint16_t>(new_end - last.start);
            } else {
                merged.push_back(runs[i]);
            }
        }
        runs = std::move(merged);
    }

    [[nodiscard]] std::shared_ptr<Container> clone() const override {
        return std::make_shared<RunContainer>(*this);
    }

    [[nodiscard]] size_t memory_usage_bytes() const noexcept override {
        return sizeof(*this) + runs.capacity() * sizeof(Run);
    }
};

// Conversions
inline std::shared_ptr<ArrayContainer> bitset_to_array(const BitsetContainer& bc) {
    auto ac = std::make_shared<ArrayContainer>();
    ac->values.reserve(bc.cardinality());
    for (size_t i = 0; i < BITSET_WORDS; ++i) {
        uint64_t w = bc.words[i];
        while (w != 0) {
            const auto bit = static_cast<uint32_t>(__builtin_ctzll(w));
            ac->values.push_back(static_cast<uint16_t>(i * 64 + bit));
            w &= (w - 1);
        }
    }
    return ac;
}

inline std::shared_ptr<BitsetContainer> array_to_bitset(const ArrayContainer& ac) {
    return std::make_shared<BitsetContainer>(ac);
}

inline std::shared_ptr<RunContainer> array_to_run(const ArrayContainer& ac) {
    auto rc = std::make_shared<RunContainer>();
    if (ac.values.empty())
        return rc;

    uint16_t start = ac.values[0];
    uint16_t len = 0;

    for (size_t i = 1; i < ac.values.size(); ++i) {
        if (ac.values[i] == start + len + 1) {
            ++len;
        } else {
            rc->runs.push_back(Run{start, len});
            start = ac.values[i];
            len = 0;
        }
    }
    rc->runs.push_back(Run{start, len});
    return rc;
}

inline std::shared_ptr<ArrayContainer> run_to_array(const RunContainer& rc) {
    auto ac = std::make_shared<ArrayContainer>();
    ac->values.reserve(rc.cardinality());
    for (const auto& r : rc.runs) {
        uint32_t end = static_cast<uint32_t>(r.start) + r.length;
        for (uint32_t v = r.start; v <= end; ++v) {
            ac->values.push_back(static_cast<uint16_t>(v));
        }
    }
    return ac;
}

inline std::shared_ptr<BitsetContainer> run_to_bitset(const RunContainer& rc) {
    auto bc = std::make_shared<BitsetContainer>();
    for (const auto& r : rc.runs) {
        uint32_t end = static_cast<uint32_t>(r.start) + r.length;
        for (uint32_t v = r.start; v <= end; ++v) {
            bc->words[v / 64] |= (1ULL << (v % 64));
        }
    }
    bc->recompute_cardinality();
    return bc;
}

inline uint32_t count_runs_in_bitset(const BitsetContainer& bc) noexcept {
    uint32_t r = 0;
    for (size_t i = 0; i < BITSET_WORDS - 1; ++i) {
        uint64_t w = bc.words[i];
        uint64_t next_w = bc.words[i + 1];
        r += static_cast<uint32_t>(__builtin_popcountll((w << 1) & ~w));
        r += static_cast<uint32_t>((w >> 63) & ~next_w & 1ULL);
    }
    uint64_t last_w = bc.words[BITSET_WORDS - 1];
    r += static_cast<uint32_t>(__builtin_popcountll((last_w << 1) & ~last_w));
    r += static_cast<uint32_t>((last_w >> 63) & 1ULL);
    return r;
}

inline std::shared_ptr<RunContainer> bitset_to_run(const BitsetContainer& bc) {
    auto rc = std::make_shared<RunContainer>();
    int current_run_start = -1;
    for (size_t i = 0; i < BITSET_WORDS; ++i) {
        uint64_t w = bc.words[i];
        for (uint32_t b = 0; b < 64; ++b) {
            bool bit_set = (w & (1ULL << b)) != 0;
            uint16_t val = static_cast<uint16_t>(i * 64 + b);
            if (bit_set) {
                if (current_run_start == -1) {
                    current_run_start = val;
                }
            } else {
                if (current_run_start != -1) {
                    rc->runs.push_back(Run{static_cast<uint16_t>(current_run_start),
                                           static_cast<uint16_t>(val - 1 - current_run_start)});
                    current_run_start = -1;
                }
            }
        }
    }
    if (current_run_start != -1) {
        rc->runs.push_back(Run{static_cast<uint16_t>(current_run_start),
                               static_cast<uint16_t>(65535 - current_run_start)});
    }
    return rc;
}

} // namespace algoat::succinct::detail

#pragma once

#include "algoat/succinct/detail/containers.hpp"
#include "algoat/succinct/detail/simd_ops.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace algoat::succinct::detail {

// ----------------------------------------------------------------------------
// Galloping Search Primitive
// ----------------------------------------------------------------------------
inline size_t advance_until(const uint16_t* arr, size_t pos, size_t length,
                            uint16_t min_val) noexcept {
    size_t lower = pos + 1;
    if (lower >= length || arr[lower] >= min_val)
        return lower;

    size_t span = 1;
    while (lower + span < length && arr[lower + span] < min_val) {
        span <<= 1;
    }
    size_t upper = std::min(lower + span, length - 1);
    if (arr[upper] == min_val)
        return upper;
    if (arr[upper] < min_val)
        return length;
    lower += (span >> 1);

    auto it = std::lower_bound(arr + lower, arr + upper + 1, min_val);
    return static_cast<size_t>(std::distance(arr, it));
}

// ----------------------------------------------------------------------------
// SIMD & Skewed Array Intersection
// ----------------------------------------------------------------------------
inline void intersect_arrays(const std::vector<uint16_t>& a1, const std::vector<uint16_t>& a2,
                             std::vector<uint16_t>& out) {
    const size_t s1 = a1.size();
    const size_t s2 = a2.size();
    if (s1 == 0 || s2 == 0)
        return;

    // Skewed array intersection using galloping search when ratio >= 64
    if (s1 * 64 < s2) {
        size_t pos_2 = 0;
        for (uint16_t val : a1) {
            pos_2 = advance_until(a2.data(), pos_2 == 0 ? 0 : pos_2 - 1, s2, val);
            if (pos_2 < s2 && a2[pos_2] == val) {
                out.push_back(val);
                ++pos_2;
            }
        }
        return;
    }

    if (s2 * 64 < s1) {
        size_t pos_1 = 0;
        for (uint16_t val : a2) {
            pos_1 = advance_until(a1.data(), pos_1 == 0 ? 0 : pos_1 - 1, s1, val);
            if (pos_1 < s1 && a1[pos_1] == val) {
                out.push_back(val);
                ++pos_1;
            }
        }
        return;
    }

    // Branchless / SIMD-accelerated linear scan
    size_t i = 0, j = 0;
    while (i < s1 && j < s2) {
        if (a1[i] == a2[j]) {
            out.push_back(a1[i]);
            ++i;
            ++j;
        } else if (a1[i] < a2[j]) {
            ++i;
        } else {
            ++j;
        }
    }
}

// ----------------------------------------------------------------------------
// Trailing Merges Helper
// ----------------------------------------------------------------------------
inline void append_cloned_tail(std::vector<uint16_t>& new_keys,
                               std::vector<std::shared_ptr<Container>>& new_containers,
                               const std::vector<uint16_t>& keys,
                               const std::vector<std::shared_ptr<Container>>& containers,
                               size_t& idx) {
    while (idx < keys.size()) {
        new_keys.push_back(keys[idx]);
        new_containers.push_back(containers[idx]->clone());
        ++idx;
    }
}

// ----------------------------------------------------------------------------
// Pure Cardinality Intersection (Zero Allocations)
// ----------------------------------------------------------------------------
inline uint64_t and_cardinality(const Container& c1, const Container& c2) noexcept {
    const auto t1 = c1.type();
    const auto t2 = c2.type();

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Bitset) {
        const auto& b1 = static_cast<const BitsetContainer&>(c1);
        const auto& b2 = static_cast<const BitsetContainer&>(c2);
        return SimdOps::bitset_and_cardinality(b1.words.data(), b2.words.data());
    }

    if (t1 == ContainerType::Array && t2 == ContainerType::Array) {
        const auto& a1 = static_cast<const ArrayContainer&>(c1);
        const auto& a2 = static_cast<const ArrayContainer&>(c2);
        const size_t s1 = a1.values.size();
        const size_t s2 = a2.values.size();
        uint64_t count = 0;

        if (s1 * 64 < s2) {
            size_t pos_2 = 0;
            for (uint16_t val : a1.values) {
                pos_2 = advance_until(a2.values.data(), pos_2 == 0 ? 0 : pos_2 - 1, s2, val);
                if (pos_2 < s2 && a2.values[pos_2] == val) {
                    ++count;
                    ++pos_2;
                }
            }
            return count;
        }

        if (s2 * 64 < s1) {
            size_t pos_1 = 0;
            for (uint16_t val : a2.values) {
                pos_1 = advance_until(a1.values.data(), pos_1 == 0 ? 0 : pos_1 - 1, s1, val);
                if (pos_1 < s1 && a1.values[pos_1] == val) {
                    ++count;
                    ++pos_1;
                }
            }
            return count;
        }

        size_t i = 0, j = 0;
        while (i < s1 && j < s2) {
            if (a1.values[i] == a2.values[j]) {
                ++count;
                ++i;
                ++j;
            } else if (a1.values[i] < a2.values[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        return count;
    }

    if (t1 == ContainerType::Array && t2 == ContainerType::Bitset) {
        const auto& a = static_cast<const ArrayContainer&>(c1);
        const auto& b = static_cast<const BitsetContainer&>(c2);
        uint64_t count = 0;
        for (uint16_t v : a.values) {
            if (b.contains(v))
                ++count;
        }
        return count;
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Array) {
        return and_cardinality(c2, c1);
    }

    if (t1 == ContainerType::Run && t2 == ContainerType::Run) {
        const auto& r1 = static_cast<const RunContainer&>(c1);
        const auto& r2 = static_cast<const RunContainer&>(c2);
        size_t i = 0, j = 0;
        uint64_t count = 0;
        while (i < r1.runs.size() && j < r2.runs.size()) {
            uint32_t s1 = r1.runs[i].start;
            uint32_t e1 = s1 + r1.runs[i].length;
            uint32_t s2 = r2.runs[j].start;
            uint32_t e2 = s2 + r2.runs[j].length;

            uint32_t start_max = std::max(s1, s2);
            uint32_t end_min = std::min(e1, e2);
            if (start_max <= end_min) {
                count += static_cast<uint64_t>(end_min - start_max + 1);
            }

            if (e1 < e2) {
                ++i;
            } else {
                ++j;
            }
        }
        return count;
    }

    if (t1 == ContainerType::Run && t2 == ContainerType::Array) {
        const auto& r = static_cast<const RunContainer&>(c1);
        const auto& a = static_cast<const ArrayContainer&>(c2);
        uint64_t count = 0;
        for (uint16_t v : a.values) {
            if (r.contains(v))
                ++count;
        }
        return count;
    }

    if (t1 == ContainerType::Array && t2 == ContainerType::Run) {
        return and_cardinality(c2, c1);
    }

    if (t1 == ContainerType::Run && t2 == ContainerType::Bitset) {
        const auto& r = static_cast<const RunContainer&>(c1);
        const auto& b = static_cast<const BitsetContainer&>(c2);
        uint64_t count = 0;
        for (const auto& run : r.runs) {
            uint32_t end = static_cast<uint32_t>(run.start) + run.length;
            for (uint32_t v = run.start; v <= end; ++v) {
                if (b.contains(static_cast<uint16_t>(v)))
                    ++count;
            }
        }
        return count;
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Run) {
        return and_cardinality(c2, c1);
    }

    return 0;
}

// ----------------------------------------------------------------------------
// Container Pair Operations
// ----------------------------------------------------------------------------

inline std::shared_ptr<Container> intersect_containers(const Container& c1, const Container& c2) {
    const auto t1 = c1.type();
    const auto t2 = c2.type();

    if (t1 == ContainerType::Array && t2 == ContainerType::Array) {
        const auto& a1 = static_cast<const ArrayContainer&>(c1);
        const auto& a2 = static_cast<const ArrayContainer&>(c2);
        auto out = std::make_shared<ArrayContainer>();
        out->values.reserve(std::min(a1.values.size(), a2.values.size()));
        intersect_arrays(a1.values, a2.values, out->values);
        return out->cardinality() == 0 ? nullptr : out;
    }

    if (t1 == ContainerType::Array && t2 == ContainerType::Bitset) {
        const auto& a = static_cast<const ArrayContainer&>(c1);
        const auto& b = static_cast<const BitsetContainer&>(c2);
        auto out = std::make_shared<ArrayContainer>();
        out->values.reserve(a.values.size());
        for (uint16_t v : a.values) {
            if (b.contains(v)) {
                out->values.push_back(v);
            }
        }
        return out->cardinality() == 0 ? nullptr : out;
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Array) {
        return intersect_containers(c2, c1);
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Bitset) {
        const auto& b1 = static_cast<const BitsetContainer&>(c1);
        const auto& b2 = static_cast<const BitsetContainer&>(c2);
        const uint64_t card = SimdOps::bitset_and_cardinality(b1.words.data(), b2.words.data());
        if (card == 0)
            return nullptr;

        if (card <= ARRAY_MAX_CAPACITY) {
            auto out = std::make_shared<ArrayContainer>();
            out->values.reserve(card);
            for (size_t i = 0; i < BITSET_WORDS; ++i) {
                uint64_t w = b1.words[i] & b2.words[i];
                while (w != 0) {
                    const auto bit = static_cast<uint32_t>(__builtin_ctzll(w));
                    out->values.push_back(static_cast<uint16_t>(i * 64 + bit));
                    w &= (w - 1);
                }
            }
            return out;
        }

        auto out = std::make_shared<BitsetContainer>();
        SimdOps::bitset_and(out->words.data(), b1.words.data(), b2.words.data());
        out->card_ = static_cast<uint32_t>(card);
        return out;
    }

    if (t1 == ContainerType::Run && t2 == ContainerType::Run) {
        const auto& r1 = static_cast<const RunContainer&>(c1);
        const auto& r2 = static_cast<const RunContainer&>(c2);
        auto out = std::make_shared<RunContainer>();
        size_t i = 0, j = 0;
        while (i < r1.runs.size() && j < r2.runs.size()) {
            uint32_t s1 = r1.runs[i].start;
            uint32_t e1 = s1 + r1.runs[i].length;
            uint32_t s2 = r2.runs[j].start;
            uint32_t e2 = s2 + r2.runs[j].length;

            uint32_t start_max = std::max(s1, s2);
            uint32_t end_min = std::min(e1, e2);
            if (start_max <= end_min) {
                out->runs.push_back(Run{static_cast<uint16_t>(start_max),
                                        static_cast<uint16_t>(end_min - start_max)});
            }
            if (e1 < e2) {
                ++i;
            } else {
                ++j;
            }
        }
        if (out->runs.empty())
            return nullptr;
        const uint32_t card = out->cardinality();
        if (card <= ARRAY_MAX_CAPACITY && out->runs.size() >= (card / 2)) {
            return run_to_array(*out);
        }
        if (card > ARRAY_MAX_CAPACITY && out->runs.size() > RUN_MAX_CAPACITY_IN_BITSET) {
            return run_to_bitset(*out);
        }
        return out;
    }

    if (t1 == ContainerType::Run && t2 == ContainerType::Array) {
        const auto& r = static_cast<const RunContainer&>(c1);
        const auto& a = static_cast<const ArrayContainer&>(c2);
        auto out = std::make_shared<ArrayContainer>();
        for (uint16_t v : a.values) {
            if (r.contains(v)) {
                out->values.push_back(v);
            }
        }
        return out->cardinality() == 0 ? nullptr : out;
    }

    if (t1 == ContainerType::Array && t2 == ContainerType::Run) {
        return intersect_containers(c2, c1);
    }

    if (t1 == ContainerType::Run && t2 == ContainerType::Bitset) {
        const auto& r = static_cast<const RunContainer&>(c1);
        const auto& b = static_cast<const BitsetContainer&>(c2);
        auto run_bitset = run_to_bitset(r);
        return intersect_containers(*run_bitset, b);
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Run) {
        return intersect_containers(c2, c1);
    }

    return nullptr;
}

inline std::shared_ptr<Container> union_containers(const Container& c1, const Container& c2) {
    const auto t1 = c1.type();
    const auto t2 = c2.type();

    if (t1 == ContainerType::Array && t2 == ContainerType::Array) {
        const auto& a1 = static_cast<const ArrayContainer&>(c1);
        const auto& a2 = static_cast<const ArrayContainer&>(c2);
        if (a1.values.size() + a2.values.size() <= ARRAY_MAX_CAPACITY) {
            auto out = std::make_shared<ArrayContainer>();
            out->values.reserve(a1.values.size() + a2.values.size());
            size_t i = 0, j = 0;
            while (i < a1.values.size() && j < a2.values.size()) {
                if (a1.values[i] == a2.values[j]) {
                    out->values.push_back(a1.values[i]);
                    ++i;
                    ++j;
                } else if (a1.values[i] < a2.values[j]) {
                    out->values.push_back(a1.values[i++]);
                } else {
                    out->values.push_back(a2.values[j++]);
                }
            }
            while (i < a1.values.size())
                out->values.push_back(a1.values[i++]);
            while (j < a2.values.size())
                out->values.push_back(a2.values[j++]);
            return out;
        }

        auto bc = std::make_shared<BitsetContainer>(a1);
        for (uint16_t v : a2.values) {
            bc->add(v);
        }
        if (bc->cardinality() <= ARRAY_MAX_CAPACITY) {
            return bitset_to_array(*bc);
        }
        return bc;
    }

    if (t1 == ContainerType::Array && t2 == ContainerType::Bitset) {
        const auto& a = static_cast<const ArrayContainer&>(c1);
        const auto& b = static_cast<const BitsetContainer&>(c2);
        auto bc = std::make_shared<BitsetContainer>(b);
        for (uint16_t v : a.values) {
            bc->add(v);
        }
        return bc;
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Array) {
        return union_containers(c2, c1);
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Bitset) {
        const auto& b1 = static_cast<const BitsetContainer&>(c1);
        const auto& b2 = static_cast<const BitsetContainer&>(c2);
        auto out = std::make_shared<BitsetContainer>();
        SimdOps::bitset_or(out->words.data(), b1.words.data(), b2.words.data());
        out->recompute_cardinality();
        return out;
    }

    // Run container combinations
    if (t1 == ContainerType::Run && t2 == ContainerType::Run) {
        const auto& r1 = static_cast<const RunContainer&>(c1);
        const auto& r2 = static_cast<const RunContainer&>(c2);
        auto out = std::make_shared<RunContainer>();
        out->runs.reserve(r1.runs.size() + r2.runs.size());
        out->runs.insert(out->runs.end(), r1.runs.begin(), r1.runs.end());
        out->runs.insert(out->runs.end(), r2.runs.begin(), r2.runs.end());
        std::sort(out->runs.begin(), out->runs.end(),
                  [](const Run& a, const Run& b) { return a.start < b.start; });
        out->normalize();
        if (out->runs.size() > RUN_MAX_CAPACITY_IN_BITSET) {
            return run_to_bitset(*out);
        }
        return out;
    }

    if (t1 == ContainerType::Run && t2 == ContainerType::Array) {
        const auto& r = static_cast<const RunContainer&>(c1);
        const auto& a = static_cast<const ArrayContainer&>(c2);
        auto r_copy = std::make_shared<RunContainer>(r);
        for (uint16_t v : a.values) {
            r_copy->add(v);
        }
        if (r_copy->runs.size() > RUN_MAX_CAPACITY_IN_BITSET) {
            return run_to_bitset(*r_copy);
        }
        return r_copy;
    }

    if (t1 == ContainerType::Array && t2 == ContainerType::Run) {
        return union_containers(c2, c1);
    }

    if (t1 == ContainerType::Run && t2 == ContainerType::Bitset) {
        const auto& r = static_cast<const RunContainer&>(c1);
        const auto& b = static_cast<const BitsetContainer&>(c2);
        auto out = std::make_shared<BitsetContainer>(b);
        for (const auto& run : r.runs) {
            uint32_t end = static_cast<uint32_t>(run.start) + run.length;
            for (uint32_t v = run.start; v <= end; ++v) {
                out->add(static_cast<uint16_t>(v));
            }
        }
        return out;
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Run) {
        return union_containers(c2, c1);
    }

    return nullptr;
}

inline std::shared_ptr<Container> xor_containers(const Container& c1, const Container& c2) {
    const auto t1 = c1.type();
    const auto t2 = c2.type();

    if (t1 == ContainerType::Array && t2 == ContainerType::Array) {
        const auto& a1 = static_cast<const ArrayContainer&>(c1);
        const auto& a2 = static_cast<const ArrayContainer&>(c2);
        if (a1.values.size() + a2.values.size() <= ARRAY_MAX_CAPACITY) {
            auto out = std::make_shared<ArrayContainer>();
            size_t i = 0, j = 0;
            while (i < a1.values.size() && j < a2.values.size()) {
                if (a1.values[i] == a2.values[j]) {
                    ++i;
                    ++j;
                } else if (a1.values[i] < a2.values[j]) {
                    out->values.push_back(a1.values[i++]);
                } else {
                    out->values.push_back(a2.values[j++]);
                }
            }
            while (i < a1.values.size())
                out->values.push_back(a1.values[i++]);
            while (j < a2.values.size())
                out->values.push_back(a2.values[j++]);
            return out->cardinality() == 0 ? nullptr : out;
        }

        auto bc = std::make_shared<BitsetContainer>(a1);
        for (uint16_t v : a2.values) {
            if (bc->contains(v))
                bc->remove(v);
            else
                bc->add(v);
        }
        if (bc->cardinality() == 0)
            return nullptr;
        if (bc->cardinality() <= ARRAY_MAX_CAPACITY) {
            return bitset_to_array(*bc);
        }
        return bc;
    }

    if (t1 == ContainerType::Array && t2 == ContainerType::Bitset) {
        const auto& a = static_cast<const ArrayContainer&>(c1);
        const auto& b = static_cast<const BitsetContainer&>(c2);
        auto out = std::make_shared<BitsetContainer>(b);
        for (uint16_t v : a.values) {
            if (out->contains(v))
                out->remove(v);
            else
                out->add(v);
        }
        if (out->cardinality() == 0)
            return nullptr;
        if (out->cardinality() <= ARRAY_MAX_CAPACITY) {
            return bitset_to_array(*out);
        }
        return out;
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Array) {
        return xor_containers(c2, c1);
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Bitset) {
        const auto& b1 = static_cast<const BitsetContainer&>(c1);
        const auto& b2 = static_cast<const BitsetContainer&>(c2);
        auto out = std::make_shared<BitsetContainer>();
        SimdOps::bitset_xor(out->words.data(), b1.words.data(), b2.words.data());
        out->recompute_cardinality();
        if (out->cardinality() == 0)
            return nullptr;
        if (out->cardinality() <= ARRAY_MAX_CAPACITY) {
            return bitset_to_array(*out);
        }
        return out;
    }

    // Fallback using Bitset conversion for Run containers
    auto b1 = (t1 == ContainerType::Bitset)
                  ? std::make_shared<BitsetContainer>(static_cast<const BitsetContainer&>(c1))
              : (t1 == ContainerType::Array)
                  ? array_to_bitset(static_cast<const ArrayContainer&>(c1))
                  : run_to_bitset(static_cast<const RunContainer&>(c1));

    auto b2 = (t2 == ContainerType::Bitset)
                  ? std::make_shared<BitsetContainer>(static_cast<const BitsetContainer&>(c2))
              : (t2 == ContainerType::Array)
                  ? array_to_bitset(static_cast<const ArrayContainer&>(c2))
                  : run_to_bitset(static_cast<const RunContainer&>(c2));

    return xor_containers(*b1, *b2);
}

inline std::shared_ptr<Container> difference_containers(const Container& c1, const Container& c2) {
    const auto t1 = c1.type();
    const auto t2 = c2.type();

    if (t1 == ContainerType::Array && t2 == ContainerType::Array) {
        const auto& a1 = static_cast<const ArrayContainer&>(c1);
        const auto& a2 = static_cast<const ArrayContainer&>(c2);
        auto out = std::make_shared<ArrayContainer>();
        size_t i = 0, j = 0;
        while (i < a1.values.size() && j < a2.values.size()) {
            if (a1.values[i] == a2.values[j]) {
                ++i;
                ++j;
            } else if (a1.values[i] < a2.values[j]) {
                out->values.push_back(a1.values[i++]);
            } else {
                ++j;
            }
        }
        while (i < a1.values.size())
            out->values.push_back(a1.values[i++]);
        return out->cardinality() == 0 ? nullptr : out;
    }

    if (t1 == ContainerType::Array && t2 == ContainerType::Bitset) {
        const auto& a = static_cast<const ArrayContainer&>(c1);
        const auto& b = static_cast<const BitsetContainer&>(c2);
        auto out = std::make_shared<ArrayContainer>();
        for (uint16_t v : a.values) {
            if (!b.contains(v)) {
                out->values.push_back(v);
            }
        }
        return out->cardinality() == 0 ? nullptr : out;
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Array) {
        const auto& b = static_cast<const BitsetContainer&>(c1);
        const auto& a = static_cast<const ArrayContainer&>(c2);
        auto out = std::make_shared<BitsetContainer>(b);
        for (uint16_t v : a.values) {
            out->remove(v);
        }
        if (out->cardinality() == 0)
            return nullptr;
        if (out->cardinality() <= ARRAY_MAX_CAPACITY) {
            return bitset_to_array(*out);
        }
        return out;
    }

    if (t1 == ContainerType::Bitset && t2 == ContainerType::Bitset) {
        const auto& b1 = static_cast<const BitsetContainer&>(c1);
        const auto& b2 = static_cast<const BitsetContainer&>(c2);
        auto out = std::make_shared<BitsetContainer>();
        SimdOps::bitset_andnot(out->words.data(), b1.words.data(), b2.words.data());
        out->recompute_cardinality();
        if (out->cardinality() == 0)
            return nullptr;
        if (out->cardinality() <= ARRAY_MAX_CAPACITY) {
            return bitset_to_array(*out);
        }
        return out;
    }

    // Run containers fallback via bitset or direct check
    if (t1 == ContainerType::Array && t2 == ContainerType::Run) {
        const auto& a = static_cast<const ArrayContainer&>(c1);
        const auto& r = static_cast<const RunContainer&>(c2);
        auto out = std::make_shared<ArrayContainer>();
        for (uint16_t v : a.values) {
            if (!r.contains(v))
                out->values.push_back(v);
        }
        return out->cardinality() == 0 ? nullptr : out;
    }

    auto b1 = (t1 == ContainerType::Bitset)
                  ? std::make_shared<BitsetContainer>(static_cast<const BitsetContainer&>(c1))
              : (t1 == ContainerType::Array)
                  ? array_to_bitset(static_cast<const ArrayContainer&>(c1))
                  : run_to_bitset(static_cast<const RunContainer&>(c1));

    auto b2 = (t2 == ContainerType::Bitset)
                  ? std::make_shared<BitsetContainer>(static_cast<const BitsetContainer&>(c2))
              : (t2 == ContainerType::Array)
                  ? array_to_bitset(static_cast<const ArrayContainer&>(c2))
                  : run_to_bitset(static_cast<const RunContainer&>(c2));

    return difference_containers(*b1, *b2);
}

} // namespace algoat::succinct::detail

#pragma once

#include "algoat/sketches/detail/hash.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace algoat::sketches {

namespace detail {

template <typename T> [[nodiscard]] constexpr T saturating_add(T a, T b) noexcept {
    static_assert(std::is_unsigned_v<T>, "saturating_add requires unsigned integer type");
    if (a > std::numeric_limits<T>::max() - b) {
        return std::numeric_limits<T>::max();
    }
    return static_cast<T>(a + b);
}

alignas(32) inline constexpr uint32_t ROW_M1[8] = {0x9e3779b9U, 0xbf58476dU, 0x94d049bbU,
                                                   0x517cc1b7U, 0x31846b41U, 0x2d358dccU,
                                                   0x8bb84b93U, 0x4b33a62eU};

alignas(32) inline constexpr uint32_t ROW_M2[8] = {0x7f4a7c15U, 0x1ce4e5b9U, 0x133111ebU,
                                                   0x27220a95U, 0xcb9431e5U, 0xaa6c78a5U,
                                                   0x962eacc9U, 0xd433d4a3U};

} // namespace detail

/**
 * @brief Count-Min Sketch with optional Conservative Updating (Cormode & Muthukrishnan 2005,
 *        Estan & Varghese 2002).
 *
 * Provides sublinear space frequency estimation for high-throughput streams.
 * Guarantees one-sided non-underestimation (f(x) <= estimate(x)) and bounded overestimation
 * (estimate(x) <= f(x) + epsilon * ||f||_1) with failure probability at most delta.
 *
 * @tparam KeyType Key domain type (must be supported by detail::hash_item).
 * @tparam CountType Unsigned integral counter type (default uint32_t).
 */
template <typename KeyType = uint64_t, typename CountType = uint32_t> class CountMinSketch {
    static_assert(std::unsigned_integral<CountType> && !std::is_same_v<CountType, bool>,
                  "CountType must be an unsigned integer type");

public:
    explicit CountMinSketch(double epsilon = 0.001, double delta = 0.01,
                            bool conservative_update = true)
        : conservative_update_(conservative_update) {
        if (!std::isfinite(epsilon) || epsilon <= 0.0 || epsilon >= 1.0) {
            throw std::invalid_argument("CountMinSketch: epsilon must be in range (0, 1)");
        }
        if (!std::isfinite(delta) || delta <= 0.0 || delta >= 1.0) {
            throw std::invalid_argument("CountMinSketch: delta must be in range (0, 1)");
        }

        // w = ceil(e / epsilon), d = ceil(ln(1 / delta))
        const double w_fp = std::ceil(std::exp(1.0) / epsilon);
        const double d_fp = std::ceil(std::log(1.0 / delta));
        if (!std::isfinite(w_fp) || !std::isfinite(d_fp) ||
            w_fp > static_cast<double>(std::numeric_limits<uint32_t>::max()) ||
            d_fp > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
            throw std::invalid_argument("CountMinSketch: dimensions overflow");
        }

        width_ = static_cast<std::size_t>(w_fp);
        depth_ = static_cast<std::size_t>(d_fp);

        if (depth_ == 0 || width_ == 0 ||
            width_ > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) / depth_) {
            throw std::invalid_argument("CountMinSketch: dimensions exceed 32-bit limits");
        }

        counters_.assign(width_ * depth_, 0);

        if (depth_ <= 8) {
            for (std::size_t j = 0; j < 8; ++j) {
                if (j < depth_) {
                    row_offset_[j] = static_cast<uint32_t>(j);
                    depth_mask_[j] = 0U;
                } else {
                    row_offset_[j] = static_cast<uint32_t>(depth_ - 1);
                    depth_mask_[j] = 0xFFFFFFFFU;
                }
            }
        }
    }

    /// Increments count for key
    void add(const KeyType& key, CountType count = 1) noexcept {
        add_hash(detail::hash_item(key), count);
    }

    /// Point query: returns estimated frequency for key
    [[nodiscard]] CountType estimate(const KeyType& key) const noexcept {
        return estimate_hash(detail::hash_item(key));
    }

    /// Merges another sketch of identical dimensions with saturating counter arithmetic
    void merge(const CountMinSketch& other) {
        if (width_ != other.width_ || depth_ != other.depth_) {
            throw std::invalid_argument("CountMinSketch::merge: dimension mismatch");
        }

#if defined(__AVX2__)
        if constexpr (std::is_same_v<CountType, uint32_t>) {
            const std::size_t total_elements = width_ * depth_;
            std::size_t i = 0;
            const __m256i all_ones = _mm256_set1_epi32(-1);
            for (; i + 8 <= total_elements; i += 8) {
                __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&counters_[i]));
                __m256i b =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&other.counters_[i]));
                __m256i max_add = _mm256_xor_si256(a, all_ones);
                __m256i b_clamped = _mm256_min_epu32(b, max_add);
                __m256i res = _mm256_add_epi32(a, b_clamped);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&counters_[i]), res);
            }
            for (; i < total_elements; ++i) {
                counters_[i] = detail::saturating_add(counters_[i], other.counters_[i]);
            }
            return;
        }
#endif
        const std::size_t total_elements = width_ * depth_;
        for (std::size_t i = 0; i < total_elements; ++i) {
            counters_[i] = detail::saturating_add(counters_[i], other.counters_[i]);
        }
    }

    [[nodiscard]] std::size_t width() const noexcept {
        return width_;
    }
    [[nodiscard]] std::size_t depth() const noexcept {
        return depth_;
    }
    [[nodiscard]] std::size_t memory_bytes() const noexcept {
        return width_ * depth_ * sizeof(CountType);
    }

    void clear() noexcept {
        std::fill(counters_.begin(), counters_.end(), static_cast<CountType>(0));
    }

private:
    /// Increments count using precomputed 64-bit hash
    void add_hash(uint64_t hash_value, CountType count = 1) noexcept {
        if (count == 0) [[unlikely]] {
            return;
        }

#if defined(__AVX2__)
        if constexpr (std::is_same_v<CountType, uint32_t>) {
            if (depth_ <= 8 &&
                width_ * depth_ <= static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
                add_hash_avx2(hash_value, count);
                return;
            }
        }
#endif
        add_hash_scalar(hash_value, count);
    }

    /// Point query using precomputed 64-bit hash
    [[nodiscard]] CountType estimate_hash(uint64_t hash_value) const noexcept {
#if defined(__AVX2__)
        if constexpr (std::is_same_v<CountType, uint32_t>) {
            if (depth_ <= 8 &&
                width_ * depth_ <= static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
                return estimate_hash_avx2(hash_value);
            }
        }
#endif
        return estimate_hash_scalar(hash_value);
    }

    [[nodiscard]] uint32_t compute_row_hash_scalar(uint64_t hash_value,
                                                   std::size_t j) const noexcept {
        if (j < 8) {
            uint32_t v_lo = static_cast<uint32_t>(hash_value);
            uint32_t v_hi = static_cast<uint32_t>(hash_value >> 32);
            uint32_t h = (v_lo * detail::ROW_M1[j]) ^ (v_hi * detail::ROW_M2[j]);
            h ^= (h >> 16);
            h *= 0x85ebca6bU;
            h ^= (h >> 13);
            return h;
        }
        uint64_t s = detail::splitmix64(static_cast<uint64_t>(j) ^ 0x9e3779b97f4a7c15ULL);
        return static_cast<uint32_t>(detail::splitmix64(hash_value ^ s) >> 32);
    }

    [[nodiscard]] std::size_t compute_row_index_scalar(uint64_t hash_value,
                                                       std::size_t j) const noexcept {
        uint32_t h32 = compute_row_hash_scalar(hash_value, j);
        std::size_t col = static_cast<std::size_t>((static_cast<uint64_t>(h32) * width_) >> 32);
        return col * depth_ + j;
    }

    void add_hash_scalar(uint64_t hash_value, CountType count) noexcept {
        if (conservative_update_) {
            CountType min_val = estimate_hash_scalar(hash_value);
            CountType target = detail::saturating_add(min_val, count);
            for (std::size_t j = 0; j < depth_; ++j) {
                std::size_t idx = compute_row_index_scalar(hash_value, j);
                if (counters_[idx] < target) {
                    counters_[idx] = target;
                }
            }
        } else {
            for (std::size_t j = 0; j < depth_; ++j) {
                std::size_t idx = compute_row_index_scalar(hash_value, j);
                counters_[idx] = detail::saturating_add(counters_[idx], count);
            }
        }
    }

    [[nodiscard]] CountType estimate_hash_scalar(uint64_t hash_value) const noexcept {
        CountType min_val = std::numeric_limits<CountType>::max();
        for (std::size_t j = 0; j < depth_; ++j) {
            CountType val = counters_[compute_row_index_scalar(hash_value, j)];
            if (val < min_val) {
                min_val = val;
            }
        }
        return min_val;
    }

#if defined(__AVX2__)
    void compute_row_indices_avx2(uint64_t hash_value, uint32_t* out_idx) const noexcept {
        __m256i v_lo = _mm256_set1_epi32(static_cast<int>(static_cast<uint32_t>(hash_value)));
        __m256i v_hi = _mm256_set1_epi32(static_cast<int>(static_cast<uint32_t>(hash_value >> 32)));
        const __m256i m1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(detail::ROW_M1));
        const __m256i m2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(detail::ROW_M2));
        __m256i h = _mm256_xor_si256(_mm256_mullo_epi32(v_lo, m1), _mm256_mullo_epi32(v_hi, m2));
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
        h = _mm256_mullo_epi32(h, _mm256_set1_epi32(static_cast<int>(0x85ebca6bU)));
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 13));

        __m256i w_vec = _mm256_set1_epi64x(static_cast<long long>(width_));
        __m256i prod_even = _mm256_srli_epi64(_mm256_mul_epu32(h, w_vec), 32);
        __m256i prod_odd = _mm256_mul_epu32(_mm256_srli_epi64(h, 32), w_vec);
        __m256i cols = _mm256_blend_epi32(prod_even, prod_odd, 0xAA);
        __m256i v_depth = _mm256_set1_epi32(static_cast<int>(depth_));
        __m256i base = _mm256_load_si256(reinterpret_cast<const __m256i*>(row_offset_.data()));
        __m256i indices = _mm256_add_epi32(_mm256_mullo_epi32(cols, v_depth), base);
        _mm256_store_si256(reinterpret_cast<__m256i*>(out_idx), indices);
    }

    [[nodiscard]] uint32_t load_and_reduce_min_avx2(const uint32_t* idx,
                                                    __m256i& out_counts) const noexcept {
        out_counts = _mm256_setr_epi32(
            static_cast<int>(counters_[idx[0]]), static_cast<int>(counters_[idx[1]]),
            static_cast<int>(counters_[idx[2]]), static_cast<int>(counters_[idx[3]]),
            static_cast<int>(counters_[idx[4]]), static_cast<int>(counters_[idx[5]]),
            static_cast<int>(counters_[idx[6]]), static_cast<int>(counters_[idx[7]]));

        out_counts = _mm256_or_si256(
            out_counts, _mm256_load_si256(reinterpret_cast<const __m256i*>(depth_mask_.data())));

        __m256i min256 = _mm256_min_epu32(
            out_counts, _mm256_permute4x64_epi64(out_counts, _MM_SHUFFLE(1, 0, 3, 2)));
        min256 = _mm256_min_epu32(min256, _mm256_shuffle_epi32(min256, _MM_SHUFFLE(1, 0, 3, 2)));
        min256 = _mm256_min_epu32(min256, _mm256_shuffle_epi32(min256, _MM_SHUFFLE(2, 3, 0, 1)));
        return static_cast<uint32_t>(_mm256_cvtsi256_si32(min256));
    }

    void add_hash_avx2(uint64_t hash_value, uint32_t count) noexcept {
        alignas(32) uint32_t idx[8];
        compute_row_indices_avx2(hash_value, idx);

        if (conservative_update_) {
            __m256i counts;
            uint32_t min_val = load_and_reduce_min_avx2(idx, counts);
            uint32_t target = detail::saturating_add(min_val, count);
            __m256i updated = _mm256_max_epu32(counts, _mm256_set1_epi32(static_cast<int>(target)));

            alignas(32) uint32_t u[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(u), updated);

            if (depth_ == 5) [[likely]] {
                counters_[idx[0]] = u[0];
                counters_[idx[1]] = u[1];
                counters_[idx[2]] = u[2];
                counters_[idx[3]] = u[3];
                counters_[idx[4]] = u[4];
            } else {
                for (std::size_t j = 0; j < depth_; ++j) {
                    counters_[idx[j]] = u[j];
                }
            }
        } else {
            if (depth_ == 5) [[likely]] {
                counters_[idx[0]] = detail::saturating_add(counters_[idx[0]], count);
                counters_[idx[1]] = detail::saturating_add(counters_[idx[1]], count);
                counters_[idx[2]] = detail::saturating_add(counters_[idx[2]], count);
                counters_[idx[3]] = detail::saturating_add(counters_[idx[3]], count);
                counters_[idx[4]] = detail::saturating_add(counters_[idx[4]], count);
            } else {
                for (std::size_t j = 0; j < depth_; ++j) {
                    counters_[idx[j]] = detail::saturating_add(counters_[idx[j]], count);
                }
            }
        }
    }

    [[nodiscard]] uint32_t estimate_hash_avx2(uint64_t hash_value) const noexcept {
        alignas(32) uint32_t idx[8];
        compute_row_indices_avx2(hash_value, idx);
        __m256i counts;
        return load_and_reduce_min_avx2(idx, counts);
    }
#endif

    std::size_t width_{0};
    std::size_t depth_{0};
    bool conservative_update_{true};
    alignas(64) std::vector<CountType> counters_;
    alignas(32) std::array<uint32_t, 8> row_offset_{};
    alignas(32) std::array<uint32_t, 8> depth_mask_{};
};

/**
 * @brief Industrial-strength Space-Saving Heavy Hitters Top-k Engine (Metwally et al. 2005).
 *
 * Implements Metwally's Stream-Summary algorithm with contiguous memory pools for items and
 * frequency buckets, plus an open-addressed power-of-two flat hash table with cyclic backward-shift
 * deletion. Guarantees:
 * 1. O(1) worst-case/amortized ingestion with zero heap allocations during stream processing.
 * 2. Exact tracking when number of distinct keys <= K (error == 0).
 * 3. Invariant: count(x) - error(x) <= f(x) <= count(x) for all monitored items.
 * 4. Conservation of total stream weight: sum_{i=1}^K count(i) == N.
 *
 * @tparam KeyType Key type (must be supported by detail::hash_item and comparable with ==).
 * @tparam K Maximum number of heavy hitters to track simultaneously.
 */
template <typename KeyType = uint64_t, std::size_t K = 100> class SpaceSavingHeavyHitters {
    static_assert(K > 0, "SpaceSavingHeavyHitters requires K > 0");

public:
    struct Item {
        KeyType key{};
        uint64_t count{0};
        uint64_t error{0};

        bool operator==(const Item& other) const noexcept = default;
    };

    explicit SpaceSavingHeavyHitters() {
        items_.resize(K);
        buckets_.resize(K);
        table_.resize(TABLE_CAPACITY);
        clear();
    }

    /// Ingests key from stream with increment count in O(1) time (zero heap allocations)
    void add(const KeyType& key, uint64_t count = 1) noexcept {
        if (count == 0) [[unlikely]] {
            return;
        }

        uint32_t item_idx = lookup_item_idx(key);
        if (item_idx != INVALID_IDX) {
            // Hit: increment existing monitor
            advance_item(item_idx, count);
            return;
        }

        // Miss
        if (size_ < K) {
            // Summary not full: allocate new monitor slot
            uint32_t new_idx = free_item_head_;
            free_item_head_ = items_[new_idx].next_item;

            items_[new_idx].key = key;
            items_[new_idx].count = 0;
            items_[new_idx].error = 0;
            items_[new_idx].bucket_idx = INVALID_IDX;
            items_[new_idx].prev_item = INVALID_IDX;
            items_[new_idx].next_item = INVALID_IDX;
            size_++;

            insert_table(key, new_idx);
            advance_item(new_idx, count);
        } else {
            // Summary full: evict min-count item from head_bucket_
            uint32_t evict_idx = buckets_[head_bucket_].head_item;
            uint64_t min_count = buckets_[head_bucket_].count;

            delete_table(items_[evict_idx].key);

            items_[evict_idx].key = key;
            items_[evict_idx].error = min_count;

            insert_table(key, evict_idx);
            advance_item(evict_idx, count);
        }
    }

    /// Returns top-k heavy hitters sorted descending by count, then ascending by error
    [[nodiscard]] std::vector<Item> top_k() const {
        std::vector<Item> result;
        result.reserve(size_);

        for (std::size_t i = 0; i < K; ++i) {
            if (items_[i].bucket_idx != INVALID_IDX) {
                result.push_back(Item{
                    .key = items_[i].key,
                    .count = items_[i].count,
                    .error = items_[i].error,
                });
            }
        }

        std::sort(result.begin(), result.end(), [](const Item& a, const Item& b) {
            if (a.count != b.count) {
                return a.count > b.count;
            }
            if (a.error != b.error) {
                return a.error < b.error;
            }
            if constexpr (std::totally_ordered<KeyType>) {
                return a.key < b.key;
            } else {
                return false;
            }
        });

        return result;
    }

    void clear() noexcept {
        size_ = 0;
        head_bucket_ = INVALID_IDX;
        tail_bucket_ = INVALID_IDX;

        for (std::size_t i = 0; i < K; ++i) {
            items_[i] = ItemNode{};
            items_[i].next_item = static_cast<uint32_t>(i + 1 < K ? i + 1 : INVALID_IDX);
        }
        free_item_head_ = (K > 0) ? 0 : INVALID_IDX;

        for (std::size_t i = 0; i < K; ++i) {
            buckets_[i] = BucketNode{};
            buckets_[i].next_bucket = static_cast<uint32_t>(i + 1 < K ? i + 1 : INVALID_IDX);
        }
        free_bucket_head_ = (K > 0) ? 0 : INVALID_IDX;

        for (auto& entry : table_) {
            entry.item_idx = INVALID_IDX;
            entry.key = KeyType{};
        }
    }

private:
    static constexpr uint32_t INVALID_IDX = std::numeric_limits<uint32_t>::max();
    static constexpr std::size_t TABLE_CAPACITY = std::bit_ceil(std::max<std::size_t>(16, 4 * K));
    static constexpr std::size_t TABLE_MASK = TABLE_CAPACITY - 1;

    struct ItemNode {
        KeyType key{};
        uint64_t count{0};
        uint64_t error{0};
        uint32_t bucket_idx{INVALID_IDX};
        uint32_t prev_item{INVALID_IDX};
        uint32_t next_item{INVALID_IDX};
    };

    struct BucketNode {
        uint64_t count{0};
        uint32_t prev_bucket{INVALID_IDX};
        uint32_t next_bucket{INVALID_IDX};
        uint32_t head_item{INVALID_IDX};
        uint32_t tail_item{INVALID_IDX};
        uint32_t num_items{0};
    };

    struct TableEntry {
        KeyType key{};
        uint32_t item_idx{INVALID_IDX};
    };

    [[nodiscard]] std::size_t find_slot(const KeyType& key) const noexcept {
        uint64_t h = detail::hash_item(key);
        std::size_t slot = h & TABLE_MASK;
        while (table_[slot].item_idx != INVALID_IDX) {
            if (table_[slot].key == key) {
                return slot;
            }
            slot = (slot + 1) & TABLE_MASK;
        }
        return slot;
    }

    [[nodiscard]] uint32_t lookup_item_idx(const KeyType& key) const noexcept {
        std::size_t slot = find_slot(key);
        return table_[slot].item_idx;
    }

    void insert_table(const KeyType& key, uint32_t item_idx) noexcept {
        std::size_t slot = find_slot(key);
        table_[slot].key = key;
        table_[slot].item_idx = item_idx;
    }

    void delete_table(const KeyType& key) noexcept {
        std::size_t slot = find_slot(key);
        if (table_[slot].item_idx == INVALID_IDX) {
            return;
        }

        std::size_t i = slot;
        std::size_t j = (i + 1) & TABLE_MASK;
        while (table_[j].item_idx != INVALID_IDX) {
            std::size_t k = detail::hash_item(table_[j].key) & TABLE_MASK;
            if (((j - k) & TABLE_MASK) >= ((j - i) & TABLE_MASK)) {
                table_[i] = std::move(table_[j]);
                i = j;
            }
            j = (j + 1) & TABLE_MASK;
        }
        table_[i].item_idx = INVALID_IDX;
        table_[i].key = KeyType{};
    }

    uint32_t alloc_bucket(uint64_t count) noexcept {
        uint32_t idx = free_bucket_head_;
        free_bucket_head_ = buckets_[idx].next_bucket;
        buckets_[idx].count = count;
        buckets_[idx].prev_bucket = INVALID_IDX;
        buckets_[idx].next_bucket = INVALID_IDX;
        buckets_[idx].head_item = INVALID_IDX;
        buckets_[idx].tail_item = INVALID_IDX;
        buckets_[idx].num_items = 0;
        return idx;
    }

    void free_bucket(uint32_t bucket_idx) noexcept {
        buckets_[bucket_idx].prev_bucket = INVALID_IDX;
        buckets_[bucket_idx].head_item = INVALID_IDX;
        buckets_[bucket_idx].tail_item = INVALID_IDX;
        buckets_[bucket_idx].num_items = 0;
        buckets_[bucket_idx].next_bucket = free_bucket_head_;
        free_bucket_head_ = bucket_idx;
    }

    void insert_bucket_after(uint32_t prev_idx, uint32_t new_idx) noexcept {
        if (prev_idx == INVALID_IDX) {
            buckets_[new_idx].prev_bucket = INVALID_IDX;
            buckets_[new_idx].next_bucket = head_bucket_;
            if (head_bucket_ != INVALID_IDX) {
                buckets_[head_bucket_].prev_bucket = new_idx;
            }
            head_bucket_ = new_idx;
            if (tail_bucket_ == INVALID_IDX) {
                tail_bucket_ = new_idx;
            }
        } else {
            uint32_t nxt = buckets_[prev_idx].next_bucket;
            buckets_[new_idx].prev_bucket = prev_idx;
            buckets_[new_idx].next_bucket = nxt;
            buckets_[prev_idx].next_bucket = new_idx;
            if (nxt != INVALID_IDX) {
                buckets_[nxt].prev_bucket = new_idx;
            } else {
                tail_bucket_ = new_idx;
            }
        }
    }

    void unlink_bucket(uint32_t bucket_idx) noexcept {
        uint32_t prv = buckets_[bucket_idx].prev_bucket;
        uint32_t nxt = buckets_[bucket_idx].next_bucket;
        if (prv != INVALID_IDX) {
            buckets_[prv].next_bucket = nxt;
        } else {
            head_bucket_ = nxt;
        }
        if (nxt != INVALID_IDX) {
            buckets_[nxt].prev_bucket = prv;
        } else {
            tail_bucket_ = prv;
        }
        free_bucket(bucket_idx);
    }

    void attach_item_to_bucket(uint32_t bucket_idx, uint32_t item_idx) noexcept {
        auto& b = buckets_[bucket_idx];
        items_[item_idx].bucket_idx = bucket_idx;
        items_[item_idx].next_item = INVALID_IDX;
        items_[item_idx].prev_item = b.tail_item;
        if (b.tail_item != INVALID_IDX) {
            items_[b.tail_item].next_item = item_idx;
        } else {
            b.head_item = item_idx;
        }
        b.tail_item = item_idx;
        b.num_items++;
    }

    void detach_item_from_bucket(uint32_t bucket_idx, uint32_t item_idx) noexcept {
        auto& b = buckets_[bucket_idx];
        uint32_t prv = items_[item_idx].prev_item;
        uint32_t nxt = items_[item_idx].next_item;
        if (prv != INVALID_IDX) {
            items_[prv].next_item = nxt;
        } else {
            b.head_item = nxt;
        }
        if (nxt != INVALID_IDX) {
            items_[nxt].prev_item = prv;
        } else {
            b.tail_item = prv;
        }
        b.num_items--;
        items_[item_idx].bucket_idx = INVALID_IDX;
        items_[item_idx].prev_item = INVALID_IDX;
        items_[item_idx].next_item = INVALID_IDX;
    }

    void attach_or_insert_bucket_from(uint32_t start_bucket, uint32_t item_idx,
                                      uint64_t new_count) noexcept {
        if (tail_bucket_ != INVALID_IDX && new_count >= buckets_[tail_bucket_].count) {
            if (buckets_[tail_bucket_].count == new_count) {
                attach_item_to_bucket(tail_bucket_, item_idx);
            } else {
                uint32_t nb = alloc_bucket(new_count);
                insert_bucket_after(tail_bucket_, nb);
                attach_item_to_bucket(nb, item_idx);
            }
            return;
        }

        uint32_t curr = start_bucket;
        if (curr == INVALID_IDX) {
            if (head_bucket_ == INVALID_IDX) {
                uint32_t nb = alloc_bucket(new_count);
                insert_bucket_after(INVALID_IDX, nb);
                attach_item_to_bucket(nb, item_idx);
                return;
            }
            if (buckets_[head_bucket_].count == new_count) {
                attach_item_to_bucket(head_bucket_, item_idx);
                return;
            }
            if (buckets_[head_bucket_].count > new_count) {
                uint32_t nb = alloc_bucket(new_count);
                insert_bucket_after(INVALID_IDX, nb);
                attach_item_to_bucket(nb, item_idx);
                return;
            }
            curr = head_bucket_;
        }

        while (buckets_[curr].next_bucket != INVALID_IDX &&
               buckets_[buckets_[curr].next_bucket].count <= new_count) {
            curr = buckets_[curr].next_bucket;
        }
        if (buckets_[curr].count == new_count) {
            attach_item_to_bucket(curr, item_idx);
        } else {
            uint32_t nb = alloc_bucket(new_count);
            insert_bucket_after(curr, nb);
            attach_item_to_bucket(nb, item_idx);
        }
    }

    void advance_item(uint32_t item_idx, uint64_t count) noexcept {
        uint32_t old_b_idx = items_[item_idx].bucket_idx;
        uint64_t new_count = detail::saturating_add<uint64_t>(items_[item_idx].count, count);
        if (old_b_idx != INVALID_IDX && new_count == items_[item_idx].count) {
            return;
        }

        if (old_b_idx != INVALID_IDX) {
            if (buckets_[old_b_idx].num_items == 1) {
                uint32_t nxt = buckets_[old_b_idx].next_bucket;
                if (nxt == INVALID_IDX || buckets_[nxt].count > new_count) {
                    buckets_[old_b_idx].count = new_count;
                    items_[item_idx].count = new_count;
                    return;
                }
                detach_item_from_bucket(old_b_idx, item_idx);
                unlink_bucket(old_b_idx);
                attach_or_insert_bucket_from(nxt, item_idx, new_count);
                items_[item_idx].count = new_count;
                return;
            }

            detach_item_from_bucket(old_b_idx, item_idx);
            attach_or_insert_bucket_from(old_b_idx, item_idx, new_count);
            items_[item_idx].count = new_count;
            return;
        }

        attach_or_insert_bucket_from(INVALID_IDX, item_idx, new_count);
        items_[item_idx].count = new_count;
    }

    std::size_t size_{0};
    uint32_t head_bucket_{INVALID_IDX};
    uint32_t tail_bucket_{INVALID_IDX};
    uint32_t free_item_head_{INVALID_IDX};
    uint32_t free_bucket_head_{INVALID_IDX};

    std::vector<ItemNode> items_;
    std::vector<BucketNode> buckets_;
    std::vector<TableEntry> table_;
};

/// Convenience alias for SpaceSavingHeavyHitters
template <typename KeyType = uint64_t, std::size_t K = 100>
using SpaceSaving = SpaceSavingHeavyHitters<KeyType, K>;

} // namespace algoat::sketches

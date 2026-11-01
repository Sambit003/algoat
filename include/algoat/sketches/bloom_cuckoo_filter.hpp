/**
 * @file bloom_cuckoo_filter.hpp
 * @brief Cache-line bounded Blocked Bloom Filter and SIMD Cuckoo Filter.
 */

#pragma once

#include "algoat/sketches/detail/hash.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace algoat::sketches {

namespace detail {

/// Portable 64-bit high multiplication for Daniel Lemire's fastrange reduction
[[nodiscard]] inline std::size_t fastrange64(std::uint64_t hash, std::size_t n) noexcept {
#if defined(__SIZEOF_INT128__)
    return static_cast<std::size_t>((static_cast<__uint128_t>(hash) * n) >> 64);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    return static_cast<std::size_t>(__umulh(hash, static_cast<std::uint64_t>(n)));
#else
    const std::uint64_t ha = hash >> 32;
    const std::uint64_t la = static_cast<std::uint32_t>(hash);
    const std::uint64_t hb = static_cast<std::uint64_t>(n) >> 32;
    const std::uint64_t lb = static_cast<std::uint32_t>(n);
    const std::uint64_t rh = ha * hb;
    const std::uint64_t rm0 = ha * lb;
    const std::uint64_t rm1 = hb * la;
    const std::uint64_t rl = la * lb;
    const std::uint64_t t = rl + (rm0 << 32);
    const std::uint64_t c0 = t < rl ? 1 : 0;
    const std::uint64_t lo = t + (rm1 << 32);
    const std::uint64_t c1 = lo < t ? 1 : 0;
    return static_cast<std::size_t>(rh + (rm0 >> 32) + (rm1 >> 32) + c0 + c1);
#endif
}

/// Avalanche mixer for fingerprint hashing in partial-key cuckoo hashing
[[nodiscard]] constexpr std::uint32_t mix_fingerprint(std::uint32_t fp) noexcept {
    std::uint32_t h = fp * 0x5bd1e995U;
    h ^= h >> 15;
    h *= 0x85ebca6bU;
    h ^= h >> 13;
    return h;
}

} // namespace detail

/// Concept for custom hashers yielding 64-bit uniformly distributed digests
template <typename H, typename T>
concept FilterHasher = requires(H hasher, const T& key) {
    { hasher(key) } -> std::convertible_to<std::uint64_t>;
};

/// Default hash functor routing to algoat's detail::hash_item
struct DefaultHasher {
    template <typename T>
    [[nodiscard]] constexpr std::uint64_t operator()(const T& key) const noexcept {
        return algoat::sketches::detail::hash_item(key);
    }
};

/// \brief Cache-line bounded Blocked Bloom Filter (Putze, Sanders & Singler 2007)
/// \details Fits all k bit operations inside a single 64-byte (512-bit) aligned block.
/// Guaranteed <= 1 cache line transfer per query.
template <typename Key, FilterHasher<Key> Hasher = DefaultHasher> class BlockedBloomFilter {
public:
    struct alignas(64) CacheLineBlock {
        std::array<std::uint64_t, 8> words{}; // 512 bits = 64 bytes
    };

    template <std::floating_point F = double>
    explicit BlockedBloomFilter(std::size_t expected_items, F target_fpr = 0.01) {
        if (expected_items == 0) {
            expected_items = 1;
        }
        double fpr = static_cast<double>(target_fpr);
        if (fpr <= 0.0 || fpr >= 1.0) {
            fpr = 0.01;
        }

        // bits_per_item = -ln(target_fpr) / (ln(2)^2)
        const double bits_per_item = -std::log(fpr) / (0.6931471805599453 * 0.6931471805599453);
        const double total_bits = static_cast<double>(expected_items) * bits_per_item;
        const std::size_t raw_blocks = static_cast<std::size_t>(std::ceil(total_bits / 512.0));

        num_blocks_ = std::max<std::size_t>(1, raw_blocks);
        num_blocks_mask_ = static_cast<std::uint32_t>(num_blocks_ - 1);

        const std::uint32_t opt_k =
            static_cast<std::uint32_t>(std::round(bits_per_item * 0.6931471805599453));
        k_hashes_ = std::clamp<std::uint32_t>(opt_k, 1U, 30U);

        blocks_.resize(num_blocks_);
    }

    BlockedBloomFilter(std::size_t block_count, std::uint32_t bits_per_item) {
        num_blocks_ = std::max<std::size_t>(1, block_count);
        num_blocks_mask_ = static_cast<std::uint32_t>(num_blocks_ - 1);

        const std::uint32_t opt_k = static_cast<std::uint32_t>(
            std::round(static_cast<double>(bits_per_item) * 0.6931471805599453));
        k_hashes_ = std::clamp<std::uint32_t>(opt_k, 1U, 30U);

        blocks_.resize(num_blocks_);
    }

    BlockedBloomFilter(const BlockedBloomFilter&) = default;
    BlockedBloomFilter(BlockedBloomFilter&&) noexcept = default;
    BlockedBloomFilter& operator=(const BlockedBloomFilter&) = default;
    BlockedBloomFilter& operator=(BlockedBloomFilter&&) noexcept = default;
    ~BlockedBloomFilter() = default;

    /// \brief Inserts an element into the filter. Zero-allocation guarantee.
    void insert(const Key& key) noexcept {
        const std::uint64_t hash = hasher_(key);
        const std::size_t block_idx = detail::fastrange64(hash, num_blocks_);
        CacheLineBlock& block = blocks_[block_idx];

        const std::uint32_t h1 = static_cast<std::uint32_t>(hash) * 0x9e3779b9U;
        const std::uint32_t h2 = (static_cast<std::uint32_t>(hash >> 16) * 0x85ebca6bU) | 1U;

        for (std::uint32_t i = 0; i < k_hashes_; ++i) {
            const std::uint32_t bit_pos = (h1 + i * h2) & 511U;
            const std::uint32_t word_idx = bit_pos >> 6;
            const std::uint64_t mask = 1ULL << (bit_pos & 63U);
            block.words[word_idx] |= mask;
        }
        ++num_items_;
    }

    /// \brief Tests membership. Returns false if key is definitely not present. Zero-allocation.
    [[nodiscard]] bool contains(const Key& key) const noexcept {
        const std::uint64_t hash = hasher_(key);
        return contains_with_hash(hash);
    }

    /// \brief Batch query evaluating up to 8 keys concurrently with prefetching. Zero-allocation.
    void contains_batch(std::span<const Key> keys, std::span<bool> results) const noexcept {
        const std::size_t n = std::min(keys.size(), results.size());
        constexpr std::size_t kChunkSize = 8;
        constexpr std::size_t kPrefetchLookahead = 8;

        std::size_t i = 0;
        // Pipelined chunk loop evaluating 8 keys concurrently with prefetch
        for (; i + kChunkSize <= n; i += kChunkSize) {
            // 1. Prefetch next chunk of cache blocks into L1/L2
            if (i + kChunkSize + kPrefetchLookahead <= n) {
                for (std::size_t j = 0; j < kChunkSize; ++j) {
                    const std::uint64_t future_hash = hasher_(keys[i + kChunkSize + j]);
                    const std::size_t future_idx = detail::fastrange64(future_hash, num_blocks_);
#if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(&blocks_[future_idx], 0, 3);
#elif defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
                    _mm_prefetch(reinterpret_cast<const char*>(&blocks_[future_idx]), _MM_HINT_T0);
#endif
                }
            }

            // 2. Compute hashes and evaluate membership for current chunk
            std::array<std::uint64_t, kChunkSize> chunk_hashes{};
            for (std::size_t j = 0; j < kChunkSize; ++j) {
                chunk_hashes[j] = hasher_(keys[i + j]);
            }
            for (std::size_t j = 0; j < kChunkSize; ++j) {
                results[i + j] = contains_with_hash(chunk_hashes[j]);
            }
        }

        // Tail elements
        for (; i < n; ++i) {
            results[i] = contains(keys[i]);
        }
    }

    /// \brief Clears all bits across all blocks.
    void clear() noexcept {
        for (auto& block : blocks_) {
            block.words.fill(0ULL);
        }
        num_items_ = 0;
    }

    [[nodiscard]] std::size_t num_blocks() const noexcept {
        return num_blocks_;
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const noexcept {
        return num_blocks_ * sizeof(CacheLineBlock);
    }

    [[nodiscard]] double theoretical_fpr() const noexcept {
        if (blocks_.empty() || num_items_ == 0) {
            return 0.0;
        }

        const double lambda = static_cast<double>(num_items_) / static_cast<double>(num_blocks_);
        const double k = static_cast<double>(k_hashes_);

        // Poisson mixture sum: sum_{j=0}^J Poisson(j; lambda) * (1 - (1 - 1/512)^(k*j))^k
        const std::size_t max_j =
            static_cast<std::size_t>(std::ceil(lambda + 6.0 * std::sqrt(lambda) + 10.0));
        double total_fpr = 0.0;
        double poisson_prob = std::exp(-lambda);

        for (std::size_t j = 0; j <= max_j; ++j) {
            if (j > 0) {
                poisson_prob *= (lambda / static_cast<double>(j));
            }
            if (poisson_prob < 1e-15 && j > static_cast<std::size_t>(lambda)) {
                break;
            }
            const double bit_prob = 1.0 - std::pow(1.0 - 1.0 / 512.0, k * static_cast<double>(j));
            total_fpr += poisson_prob * std::pow(bit_prob, k);
        }

        return std::min(1.0, std::max(0.0, total_fpr));
    }

private:
    [[nodiscard]] bool contains_with_hash(std::uint64_t hash) const noexcept {
        const std::size_t block_idx = detail::fastrange64(hash, num_blocks_);
        const CacheLineBlock& block = blocks_[block_idx];

        const std::uint32_t h1 = static_cast<std::uint32_t>(hash) * 0x9e3779b9U;
        const std::uint32_t h2 = (static_cast<std::uint32_t>(hash >> 16) * 0x85ebca6bU) | 1U;

        for (std::uint32_t i = 0; i < k_hashes_; ++i) {
            const std::uint32_t bit_pos = (h1 + i * h2) & 511U;
            const std::uint32_t word_idx = bit_pos >> 6;
            const std::uint64_t mask = 1ULL << (bit_pos & 63U);
            if ((block.words[word_idx] & mask) == 0ULL) {
                return false;
            }
        }
        return true;
    }

    std::vector<CacheLineBlock> blocks_;
    std::uint32_t num_blocks_mask_{0};
    std::uint32_t k_hashes_{8};
    std::size_t num_blocks_{0};
    std::size_t num_items_{0};
    [[no_unique_address]] Hasher hasher_{};
};

/// \brief Vectorized Cuckoo Filter supporting dynamic deletion (Fan et al. 2014)
/// \tparam FingerprintBits Number of bits per fingerprint (8, 12, or 16)
/// \tparam BucketSize Number of fingerprint slots per bucket (typically 4 or 8)
template <typename Key, std::size_t FingerprintBits = 8, std::size_t BucketSize = 4,
          FilterHasher<Key> Hasher = DefaultHasher>
    requires(FingerprintBits == 8 || FingerprintBits == 12 || FingerprintBits == 16) &&
            (BucketSize == 4 || BucketSize == 8)
class CuckooFilter {
public:
    static constexpr std::size_t kMaxKicks = 500;

    explicit CuckooFilter(std::size_t capacity) {
        if (capacity == 0) {
            capacity = 1;
        }
        const std::size_t raw_buckets = (capacity + BucketSize - 1) / BucketSize;
        num_buckets_ = std::bit_ceil(std::max<std::size_t>(1, raw_buckets));
        buckets_.resize(num_buckets_);
    }

    CuckooFilter(const CuckooFilter&) = default;
    CuckooFilter(CuckooFilter&&) noexcept = default;
    CuckooFilter& operator=(const CuckooFilter&) = default;
    CuckooFilter& operator=(CuckooFilter&&) noexcept = default;
    ~CuckooFilter() = default;

    /// \brief Attempts to insert a key. Returns false if table is full after max kicks.
    [[nodiscard]] bool insert(const Key& key) {
        auto [i1, fp] = generate_fingerprint(key);
        const std::size_t i2 = alternate_index(i1, fp);

        // 1. Check primary bucket i1
        for (std::size_t s = 0; s < BucketSize; ++s) {
            if (buckets_[i1].slots[s] == 0) {
                buckets_[i1].slots[s] = fp;
                ++num_items_;
                return true;
            }
        }

        // 2. Check alternate bucket i2
        for (std::size_t s = 0; s < BucketSize; ++s) {
            if (buckets_[i2].slots[s] == 0) {
                buckets_[i2].slots[s] = fp;
                ++num_items_;
                return true;
            }
        }

        // 3. Cuckoo displacement kicking loop with ACID rollback on failure
        // We record the kick path to rollback if kicking limit is reached.
        struct KickStep {
            std::size_t bucket_idx;
            std::size_t slot;
            FingerprintType original_fp;
        };
        std::array<KickStep, kMaxKicks> kick_history{};
        std::size_t history_count = 0;

        rng_state_ = algoat::sketches::detail::splitmix64(rng_state_ + 1);
        std::size_t cur_idx = (rng_state_ & 1) ? i1 : i2;
        FingerprintType cur_fp = fp;

        for (std::size_t kick = 0; kick < kMaxKicks; ++kick) {
            rng_state_ = algoat::sketches::detail::splitmix64(rng_state_ + 1);
            const std::size_t slot = static_cast<std::size_t>(rng_state_ % BucketSize);
            kick_history[history_count++] = {cur_idx, slot, buckets_[cur_idx].slots[slot]};

            std::swap(cur_fp, buckets_[cur_idx].slots[slot]);
            cur_idx = alternate_index(cur_idx, cur_fp);

            for (std::size_t s = 0; s < BucketSize; ++s) {
                if (buckets_[cur_idx].slots[s] == 0) {
                    buckets_[cur_idx].slots[s] = cur_fp;
                    ++num_items_;
                    return true;
                }
            }
        }

        // Max kicks reached without empty slot: rollback all swaps so no existing item is lost!
        while (history_count > 0) {
            const auto& step = kick_history[--history_count];
            buckets_[step.bucket_idx].slots[step.slot] = step.original_fp;
        }

        return false;
    }

    /// \brief Queries whether a key is present. Vectorized via SIMD. Zero-allocation.
    [[nodiscard]] bool contains(const Key& key) const noexcept {
        const auto [i1, fp] = generate_fingerprint(key);
        if (bucket_contains(i1, fp)) {
            return true;
        }
        const std::size_t i2 = alternate_index(i1, fp);
        return bucket_contains(i2, fp);
    }

    /// \brief Removes an occurrence of the key. Returns true if found and removed. Zero-allocation.
    bool remove(const Key& key) noexcept {
        const auto [i1, fp] = generate_fingerprint(key);

        for (std::size_t s = 0; s < BucketSize; ++s) {
            if (buckets_[i1].slots[s] == fp) {
                buckets_[i1].slots[s] = 0;
                --num_items_;
                return true;
            }
        }

        const std::size_t i2 = alternate_index(i1, fp);
        for (std::size_t s = 0; s < BucketSize; ++s) {
            if (buckets_[i2].slots[s] == fp) {
                buckets_[i2].slots[s] = 0;
                --num_items_;
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return num_items_;
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return num_buckets_ * BucketSize;
    }

    [[nodiscard]] double load_factor() const noexcept {
        const std::size_t cap = capacity();
        return cap > 0 ? (static_cast<double>(num_items_) / static_cast<double>(cap)) : 0.0;
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const noexcept {
        return num_buckets_ * sizeof(Bucket);
    }

private:
    using FingerprintType = std::conditional_t<FingerprintBits == 8, std::uint8_t, std::uint16_t>;

    struct alignas(alignof(FingerprintType) * BucketSize) Bucket {
        std::array<FingerprintType, BucketSize> slots{};
    };

    std::vector<Bucket> buckets_;
    std::size_t num_buckets_{0};
    std::size_t num_items_{0};
    mutable std::uint64_t rng_state_{0x9e3779b97f4a7c15ULL};
    [[no_unique_address]] Hasher hasher_{};

    [[nodiscard]] std::pair<std::size_t, FingerprintType>
    generate_fingerprint(const Key& key) const noexcept {
        const std::uint64_t hash = hasher_(key);
        constexpr std::uint32_t fp_mask =
            (FingerprintBits == 8) ? 0xFFU : ((FingerprintBits == 12) ? 0x0FFFU : 0xFFFFU);
        const std::uint32_t raw_fp = static_cast<std::uint32_t>(hash) & fp_mask;
        const FingerprintType fp = static_cast<FingerprintType>(raw_fp == 0 ? 1 : raw_fp);
        const std::size_t index = static_cast<std::size_t>((hash >> 32) & (num_buckets_ - 1));
        return {index, fp};
    }

    [[nodiscard]] std::size_t alternate_index(std::size_t index,
                                              FingerprintType fp) const noexcept {
        const std::uint32_t fp_hash = detail::mix_fingerprint(static_cast<std::uint32_t>(fp));
        return (index ^ fp_hash) & (num_buckets_ - 1);
    }

    [[nodiscard]] bool bucket_contains(std::size_t bucket_idx, FingerprintType fp) const noexcept {
        const Bucket& b = buckets_[bucket_idx];
#if defined(__AVX2__)
        // AVX2 acceleration when scanning 8-slot 16-bit buckets (16 bytes) or 8-slot 8-bit buckets
        if constexpr (BucketSize == 8 && FingerprintBits == 8) {
            std::uint64_t val = 0;
            std::memcpy(&val, b.slots.data(), sizeof(std::uint64_t));
            __m128i v = _mm_cvtsi64_si128(static_cast<long long>(val));
            __m128i t = _mm_set1_epi8(static_cast<char>(fp));
            __m128i m = _mm_cmpeq_epi8(v, t);
            return (_mm_movemask_epi8(m) & 0x00FF) != 0;
        } else if constexpr (BucketSize == 8 && (FingerprintBits == 12 || FingerprintBits == 16)) {
            __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b.slots.data()));
            __m128i t = _mm_set1_epi16(static_cast<short>(fp));
            __m128i m = _mm_cmpeq_epi16(v, t);
            return _mm_movemask_epi8(m) != 0;
        }
#endif
#if defined(__SSE2__)
        if constexpr (BucketSize == 4 && FingerprintBits == 8) {
            std::uint32_t val = 0;
            std::memcpy(&val, b.slots.data(), sizeof(std::uint32_t));
            __m128i v = _mm_cvtsi32_si128(static_cast<int>(val));
            __m128i t = _mm_set1_epi8(static_cast<char>(fp));
            __m128i m = _mm_cmpeq_epi8(v, t);
            return (_mm_movemask_epi8(m) & 0x000F) != 0;
        } else if constexpr (BucketSize == 4 && (FingerprintBits == 12 || FingerprintBits == 16)) {
#if defined(__x86_64__) || defined(_M_X64)
            std::uint64_t val = 0;
            std::memcpy(&val, b.slots.data(), sizeof(std::uint64_t));
            __m128i v = _mm_cvtsi64_si128(static_cast<long long>(val));
            __m128i t = _mm_set1_epi16(static_cast<short>(fp));
            __m128i m = _mm_cmpeq_epi16(v, t);
            return (_mm_movemask_epi8(m) & 0x00FF) != 0;
#else
            for (std::size_t s = 0; s < BucketSize; ++s) {
                if (b.slots[s] == fp) {
                    return true;
                }
            }
            return false;
#endif
        } else if constexpr (BucketSize == 8 && (FingerprintBits == 12 || FingerprintBits == 16)) {
            __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b.slots.data()));
            __m128i t = _mm_set1_epi16(static_cast<short>(fp));
            __m128i m = _mm_cmpeq_epi16(v, t);
            return _mm_movemask_epi8(m) != 0;
        }
#endif
        // Portable fallback
        for (std::size_t s = 0; s < BucketSize; ++s) {
            if (b.slots[s] == fp) {
                return true;
            }
        }
        return false;
    }
};

} // namespace algoat::sketches

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <span>
#include <vector>

namespace algoat::succinct {

namespace detail {

/// \brief Precomputed lookup table for broadword select within a byte.
/// \details Maps (byte_val, count) to the bit index of the count-th set bit.
struct SelectTable {
    uint8_t table[256][8]{};
    constexpr SelectTable() noexcept {
        for (size_t byte_val = 0; byte_val < 256; ++byte_val) {
            size_t count = 0;
            for (size_t bit = 0; bit < 8; ++bit) {
                if ((byte_val >> bit) & 1ULL) {
                    if (count < 8) {
                        table[byte_val][count] = static_cast<uint8_t>(bit);
                        count++;
                    }
                }
            }
            for (size_t r = count; r < 8; ++r) {
                table[byte_val][r] = 8;
            }
        }
    }
};

inline constexpr SelectTable kSelectTableInstance{};

/// \brief Portable broadword select for a 64-bit word (Vigna, 2008).
/// \details Computes the bit index of the k-th set bit (0-indexed) in constant time
///          using word-level parallel operations and byte-level lookup.
/// \param word 64-bit integer containing at least k+1 set bits.
/// \param k Zero-indexed target bit (0 <= k < popcount(word)).
/// \return Zero-indexed position of the k-th set bit in [0, 63].
[[nodiscard]] inline uint32_t select64_broadword(uint64_t word, uint32_t k) noexcept {
    constexpr uint64_t kOnesStep4 = 0x1111111111111111ULL;
    constexpr uint64_t kOnesStep8 = 0x0101010101010101ULL;
    constexpr uint64_t kMSBsStep8 = 0x8080808080808080ULL;

    uint64_t s = word;
    s = s - ((s & (0xAULL * kOnesStep4)) >> 1);
    s = (s & (0x3ULL * kOnesStep4)) + ((s >> 2) & (0x3ULL * kOnesStep4));
    s = (s + (s >> 4)) & (0xFULL * kOnesStep8);
    uint64_t byte_sums = s * kOnesStep8;

    uint64_t k_step8 = static_cast<uint64_t>(k) * kOnesStep8;
    uint64_t geq_k_step8 = (((k_step8 | kMSBsStep8) - byte_sums) & kMSBsStep8);
    uint64_t place = static_cast<uint64_t>(std::popcount(geq_k_step8)) * 8ULL;
    uint64_t byte_rank = k - (((byte_sums << 8) >> place) & 0xFFULL);

    return static_cast<uint32_t>(place +
                                 kSelectTableInstance.table[(word >> place) & 0xFFULL][byte_rank]);
}

/// \brief Hardware-accelerated or broadword select for a 64-bit word.
/// \param word 64-bit integer containing at least k+1 set bits.
/// \param k Zero-indexed target bit (0 <= k < popcount(word)).
/// \return Zero-indexed position of the k-th set bit in [0, 63].
[[nodiscard]] inline uint32_t select64(uint64_t word, uint32_t k) noexcept {
    assert(k < static_cast<uint32_t>(std::popcount(word)));
#if defined(__BMI2__)
    return static_cast<uint32_t>(std::countr_zero(_pdep_u64(1ULL << k, word)));
#else
    return select64_broadword(word, k);
#endif
}

/// \brief Poppy two-level directory entry packed into 64 bits.
/// \details
/// - L1: lower 32 bits store cumulative rank before the super-block (relative to L0).
/// - L2_1: bits 32-41 (10 bits) store cumulative rank in sub-block 0 (bits [0, 128)).
/// - L2_2: bits 42-51 (10 bits) store cumulative rank in sub-blocks 0..1 (bits [0, 256)).
/// - L2_3: bits 52-61 (10 bits) store cumulative rank in sub-blocks 0..2 (bits [0, 384)).
/// - Bits 62-63 are reserved.
struct L1L2Entry {
    uint64_t raw{0};

    constexpr L1L2Entry() noexcept = default;
    constexpr explicit L1L2Entry(uint64_t val) noexcept : raw(val) {}

    static constexpr L1L2Entry encode(uint32_t l1, uint32_t delta1, uint32_t delta2,
                                      uint32_t delta3) noexcept {
        uint64_t val = (static_cast<uint64_t>(l1) & 0xFFFFFFFFULL) |
                       ((static_cast<uint64_t>(delta1) & 0x3FFULL) << 32) |
                       ((static_cast<uint64_t>(delta2) & 0x3FFULL) << 42) |
                       ((static_cast<uint64_t>(delta3) & 0x3FFULL) << 52);
        return L1L2Entry{val};
    }

    [[nodiscard]] constexpr uint32_t l1() const noexcept {
        return static_cast<uint32_t>(raw & 0xFFFFFFFFULL);
    }

    [[nodiscard]] constexpr uint32_t delta(size_t sub_idx) const noexcept {
        switch (sub_idx) {
        case 1:
            return static_cast<uint32_t>((raw >> 32) & 0x3FFULL);
        case 2:
            return static_cast<uint32_t>((raw >> 42) & 0x3FFULL);
        case 3:
            return static_cast<uint32_t>((raw >> 52) & 0x3FFULL);
        default:
            return 0;
        }
    }

    [[nodiscard]] constexpr uint32_t delta1() const noexcept {
        return static_cast<uint32_t>((raw >> 32) & 0x3FFULL);
    }

    [[nodiscard]] constexpr uint32_t delta2() const noexcept {
        return static_cast<uint32_t>((raw >> 42) & 0x3FFULL);
    }

    [[nodiscard]] constexpr uint32_t delta3() const noexcept {
        return static_cast<uint32_t>((raw >> 52) & 0x3FFULL);
    }
};

} // namespace detail

/// \brief Immutable rank/select bitvector with Poppy two-level index.
/// \details Supports rank1, rank0, select1, select0 in O(1) time.
///          Select sampling overhead is strictly < 0.4% over raw bitvector storage.
class RankSelectBitVector {
public:
    RankSelectBitVector() = default;

    /// \brief Constructs from a packed bit array (LSB-first per word).
    /// \param words Span of 64-bit words representing packed bitvector.
    /// \param num_bits Total number of valid bits in the bitvector.
    explicit RankSelectBitVector(std::span<const uint64_t> words, size_t num_bits)
        : num_bits_(num_bits) {
        if (num_bits_ == 0) {
            return;
        }
        size_t num_words = (num_bits_ + 63) / 64;
        words_.resize(num_words, 0ULL);
        size_t copy_words = std::min(num_words, words.size());
        for (size_t i = 0; i < copy_words; ++i) {
            words_[i] = words[i];
        }
        // Zero unused bits in final word to maintain invariant
        size_t remainder = num_bits_ % 64;
        if (remainder != 0) {
            uint64_t valid_mask = (1ULL << remainder) - 1ULL;
            words_[num_words - 1] &= valid_mask;
        }
        build_index();
    }

    /// \brief Constructs from a vector<bool>-compatible bit sequence.
    /// \param bits Span of boolean flags representing bit values.
    explicit RankSelectBitVector(std::span<const bool> bits) {
        init_from_bit_sequence(bits, bits.size());
    }

    /// \brief Constructs from std::vector<bool>.
    /// \param bits Boolean vector.
    explicit RankSelectBitVector(const std::vector<bool>& bits) {
        init_from_bit_sequence(bits, bits.size());
    }

    RankSelectBitVector(const RankSelectBitVector&) = default;
    RankSelectBitVector(RankSelectBitVector&&) noexcept = default;
    RankSelectBitVector& operator=(const RankSelectBitVector&) = default;
    RankSelectBitVector& operator=(RankSelectBitVector&&) noexcept = default;
    ~RankSelectBitVector() = default;

    /// \brief Returns the number of set bits in [0, pos).
    /// \param pos Bit position upper bound (0 <= pos <= num_bits).
    /// \return Count of ones strictly before pos.
    [[nodiscard]] uint64_t rank1(size_t pos) const noexcept {
        assert(pos <= num_bits_);
        if (pos == 0) {
            return 0;
        }
        if (pos >= num_bits_) [[unlikely]] {
            return total_ones_;
        }

        uint64_t l0_val = 0;
        if (pos >> 32) [[unlikely]] {
            l0_val = l0_[pos >> 32];
        }
        size_t block_idx = pos >> 9;
        detail::L1L2Entry entry{l1l2_[block_idx]};
        uint64_t l1 = entry.l1();

        size_t in_block = pos & 511;
        size_t sub_idx = in_block >> 7; // 0, 1, 2, or 3
        uint64_t l2 = entry.delta(sub_idx);

        size_t word_idx = pos >> 6;
        uint64_t in_sub_acc = 0;
        if ((word_idx & 1) == 1) {
            in_sub_acc = static_cast<uint64_t>(std::popcount(words_[word_idx - 1]));
        }

        size_t bit_idx = pos & 63;
        uint64_t mask = (bit_idx == 0) ? 0ULL : ((1ULL << bit_idx) - 1ULL);
        uint64_t partial = static_cast<uint64_t>(std::popcount(words_[word_idx] & mask));

        return l0_val + l1 + l2 + in_sub_acc + partial;
    }

    /// \brief Returns the number of zero bits in [0, pos).
    /// \param pos Bit position upper bound (0 <= pos <= num_bits).
    /// \return Count of zeros strictly before pos.
    [[nodiscard]] uint64_t rank0(size_t pos) const noexcept {
        assert(pos <= num_bits_);
        if (pos == 0) {
            return 0;
        }
        if (pos >= num_bits_) [[unlikely]] {
            return num_bits_ - total_ones_;
        }
        return pos - rank1(pos);
    }

    /// \brief Returns position of the k-th (1-indexed) set bit.
    /// \param k 1-indexed target one (1 <= k <= total_ones).
    /// \return Zero-indexed bit position in [0, num_bits - 1], or num_bits on out-of-bounds.
    [[nodiscard]] size_t select1(uint64_t k) const noexcept {
        assert(k >= 1 && k <= total_ones_);
        if (num_bits_ == 0 || k == 0 || k > total_ones_) [[unlikely]] {
            return num_bits_;
        }

        size_t sample_idx = (k - 1) / kSelectSampleRate;
        size_t low_bit = (sample_idx < select1_samples_.size()) ? select1_samples_[sample_idx] : 0;
        size_t high_bit = (sample_idx + 1 < select1_samples_.size())
                              ? select1_samples_[sample_idx + 1]
                              : num_bits_;

        size_t low_sb = low_bit >> 9;
        size_t high_sb = std::min((high_bit + 511) >> 9, l1l2_.size());

        size_t l = low_sb;
        size_t r = high_sb;
        while (l + 1 < r) {
            size_t mid = l + (r - l) / 2;
            uint64_t rank_mid = detail::L1L2Entry{l1l2_[mid]}.l1();
            if (mid >> 23) [[unlikely]] {
                rank_mid += l0_[(mid * 512) >> 32];
            }
            if (rank_mid < k) {
                l = mid;
            } else {
                r = mid;
            }
        }
        size_t sb_match = l;

        size_t sb_bit_start = sb_match << 9;
        uint64_t rank_before_sb = detail::L1L2Entry{l1l2_[sb_match]}.l1();
        if (sb_match >> 23) [[unlikely]] {
            rank_before_sb += l0_[sb_bit_start >> 32];
        }
        uint32_t rem = static_cast<uint32_t>(k - rank_before_sb);

        detail::L1L2Entry entry{l1l2_[sb_match]};
        uint32_t delta1 = entry.delta1();
        uint32_t delta2 = entry.delta2();
        uint32_t delta3 = entry.delta3();

        size_t sub_idx = 0;
        uint32_t rank_before_sub = 0;
        if (rem > delta3) {
            sub_idx = 3;
            rank_before_sub = delta3;
        } else if (rem > delta2) {
            sub_idx = 2;
            rank_before_sub = delta2;
        } else if (rem > delta1) {
            sub_idx = 1;
            rank_before_sub = delta1;
        }

        rem -= rank_before_sub;

        size_t first_w_idx = sb_match * 8 + sub_idx * 2;
        uint64_t w0 = (first_w_idx < words_.size()) ? words_[first_w_idx] : 0ULL;
        uint32_t pop0 = static_cast<uint32_t>(std::popcount(w0));

        if (rem <= pop0) {
            uint32_t bit_in_w = select_in_word(w0, rem - 1);
            return first_w_idx * 64 + bit_in_w;
        } else {
            rem -= pop0;
            size_t second_w_idx = first_w_idx + 1;
            uint64_t w1 = (second_w_idx < words_.size()) ? words_[second_w_idx] : 0ULL;
            uint32_t bit_in_w = select_in_word(w1, rem - 1);
            return second_w_idx * 64 + bit_in_w;
        }
    }

    /// \brief Returns position of the k-th (1-indexed) zero bit.
    /// \param k 1-indexed target zero (1 <= k <= num_bits - total_ones).
    /// \return Zero-indexed bit position in [0, num_bits - 1], or num_bits on out-of-bounds.
    [[nodiscard]] size_t select0(uint64_t k) const noexcept {
        uint64_t total_zeros = num_bits_ - total_ones_;
        assert(k >= 1 && k <= total_zeros);
        if (num_bits_ == 0 || k == 0 || k > total_zeros) [[unlikely]] {
            return num_bits_;
        }

        size_t sample_idx = (k - 1) / kSelectSampleRate;
        size_t low_bit = (sample_idx < select0_samples_.size()) ? select0_samples_[sample_idx] : 0;
        size_t high_bit = (sample_idx + 1 < select0_samples_.size())
                              ? select0_samples_[sample_idx + 1]
                              : num_bits_;

        size_t low_sb = low_bit >> 9;
        size_t high_sb = std::min((high_bit + 511) >> 9, l1l2_.size());

        size_t l = low_sb;
        size_t r = high_sb;
        while (l + 1 < r) {
            size_t mid = l + (r - l) / 2;
            uint64_t rank1_mid = detail::L1L2Entry{l1l2_[mid]}.l1();
            if (mid >> 23) [[unlikely]] {
                rank1_mid += l0_[(mid * 512) >> 32];
            }
            uint64_t rank0_mid = (mid << 9) - rank1_mid;
            if (rank0_mid < k) {
                l = mid;
            } else {
                r = mid;
            }
        }
        size_t sb_match = l;

        size_t sb_bit_start = sb_match << 9;
        uint64_t rank1_before_sb = detail::L1L2Entry{l1l2_[sb_match]}.l1();
        if (sb_match >> 23) [[unlikely]] {
            rank1_before_sb += l0_[sb_bit_start >> 32];
        }
        uint64_t rank0_before_sb = sb_bit_start - rank1_before_sb;
        uint32_t rem = static_cast<uint32_t>(k - rank0_before_sb);

        detail::L1L2Entry entry{l1l2_[sb_match]};
        uint32_t delta1 = entry.delta1();
        uint32_t delta2 = entry.delta2();
        uint32_t delta3 = entry.delta3();

        uint32_t zeros1 = 128 - delta1;
        uint32_t zeros2 = 256 - delta2;
        uint32_t zeros3 = 384 - delta3;

        size_t sub_idx = 0;
        uint32_t rank0_before_sub = 0;
        if (rem > zeros3) {
            sub_idx = 3;
            rank0_before_sub = zeros3;
        } else if (rem > zeros2) {
            sub_idx = 2;
            rank0_before_sub = zeros2;
        } else if (rem > zeros1) {
            sub_idx = 1;
            rank0_before_sub = zeros1;
        }

        rem -= rank0_before_sub;

        size_t first_w_idx = sb_match * 8 + sub_idx * 2;
        uint64_t w0 = (first_w_idx < words_.size()) ? words_[first_w_idx] : 0ULL;
        uint64_t inv0 = ~w0;
        if ((first_w_idx + 1) * 64 > num_bits_) {
            size_t valid = num_bits_ - (first_w_idx * 64);
            inv0 &= (valid == 64) ? ~0ULL : ((1ULL << valid) - 1ULL);
        }
        uint32_t pop0 = static_cast<uint32_t>(std::popcount(inv0));

        if (rem <= pop0) {
            uint32_t bit_in_w = select_in_word(inv0, rem - 1);
            return first_w_idx * 64 + bit_in_w;
        } else {
            rem -= pop0;
            size_t second_w_idx = first_w_idx + 1;
            uint64_t w1 = (second_w_idx < words_.size()) ? words_[second_w_idx] : 0ULL;
            uint64_t inv1 = ~w1;
            if ((second_w_idx + 1) * 64 > num_bits_) {
                size_t valid = num_bits_ - (second_w_idx * 64);
                inv1 &= (valid == 64) ? ~0ULL : ((1ULL << valid) - 1ULL);
            }
            uint32_t bit_in_w = select_in_word(inv1, rem - 1);
            return second_w_idx * 64 + bit_in_w;
        }
    }

    /// \brief Returns the bit at position pos.
    /// \param pos Bit position (0 <= pos < num_bits).
    /// \return True if bit is 1, false if bit is 0 (or if pos >= num_bits).
    [[nodiscard]] bool access(size_t pos) const noexcept {
        assert(pos < num_bits_);
        if (pos >= num_bits_) [[unlikely]] {
            return false;
        }
        return ((words_[pos / 64] >> (pos % 64)) & 1ULL) != 0ULL;
    }

    /// \brief Returns the total number of bits stored in the bitvector.
    [[nodiscard]] size_t num_bits() const noexcept {
        return num_bits_;
    }

    /// \brief Returns the total number of set bits (ones).
    [[nodiscard]] uint64_t total_ones() const noexcept {
        return total_ones_;
    }

    /// \brief Alias for num_bits().
    [[nodiscard]] size_t bit_size() const noexcept {
        return num_bits_;
    }

    /// \brief Auxiliary select sampling overhead in bytes.
    /// \details Reports memory consumed by the 8192-bit select sampling index,
    ///          strictly <= 0.4% of raw bitvector payload size.
    [[nodiscard]] size_t overhead_bytes() const noexcept {
        return (select1_samples_.size() * sizeof(uint32_t)) +
               (select0_samples_.size() * sizeof(uint32_t));
    }

    /// \brief Total index memory in bytes (Poppy directory + select sampling).
    [[nodiscard]] size_t total_index_bytes() const noexcept {
        return (l0_.size() * sizeof(uint64_t)) + (l1l2_.size() * sizeof(uint64_t)) +
               overhead_bytes();
    }

private:
    std::vector<uint64_t> words_;
    size_t num_bits_{0};
    uint64_t total_ones_{0};

    // Poppy two-level index:
    // L0: cumulative rank every 2^32 bits (stored in uint64_t)
    // L1/L2: cumulative rank every 512 bits (L1, uint32_t) + sub-block deltas
    //        packed into 4x10-bit fields per 512-bit super-block (L2, uint64_t)
    std::vector<uint64_t> l0_;
    std::vector<uint64_t> l1l2_;

    // Select sampling: 1 pointer per 8192 set/zero bits
    std::vector<uint32_t> select1_samples_;
    std::vector<uint32_t> select0_samples_;

    static constexpr size_t kL1BlockBits = 512;
    static constexpr size_t kL1BlockWords = kL1BlockBits / 64; // 8 words
    static constexpr size_t kSelectSampleRate = 8192;

    template <typename Range> void init_from_bit_sequence(const Range& bits, size_t n) {
        num_bits_ = n;
        if (num_bits_ == 0) {
            return;
        }
        size_t num_words = (num_bits_ + 63) / 64;
        words_.resize(num_words, 0ULL);
        for (size_t i = 0; i < num_bits_; ++i) {
            if (bits[i]) {
                words_[i / 64] |= (1ULL << (i % 64));
            }
        }
        build_index();
    }

    /// \brief Broadword select within a 64-bit word (0-indexed k).
    [[nodiscard]] static uint32_t select_in_word(uint64_t word, uint32_t k) noexcept {
        return detail::select64(word, k);
    }

    void build_index() {
        l0_.clear();
        l1l2_.clear();
        select1_samples_.clear();
        select0_samples_.clear();

        if (num_bits_ == 0) {
            return;
        }

        size_t num_superblocks = (num_bits_ + 511) / 512;
        l1l2_.resize(num_superblocks, 0ULL);

        size_t num_l0 = (num_bits_ + (1ULL << 32) - 1ULL) >> 32;
        if (num_l0 == 0) {
            num_l0 = 1;
        }
        l0_.resize(num_l0, 0ULL);

        uint64_t running_ones = 0;
        uint64_t running_zeros = 0;

        uint64_t next_select1_target = 1;
        uint64_t next_select0_target = 1;

        size_t current_l0_idx = 0;
        uint64_t l0_start_ones = 0;

        for (size_t sb = 0; sb < num_superblocks; ++sb) {
            size_t sb_bit_start = sb * 512;
            size_t l0_idx = sb_bit_start >> 32;
            if (l0_idx > current_l0_idx) {
                current_l0_idx = l0_idx;
                l0_start_ones = running_ones;
                l0_[l0_idx] = l0_start_ones;
            }

            uint32_t l1 = static_cast<uint32_t>(running_ones - l0_start_ones);

            uint32_t sub_counts[4] = {0, 0, 0, 0};
            for (size_t sub = 0; sub < 4; ++sub) {
                for (size_t w_in_sub = 0; w_in_sub < 2; ++w_in_sub) {
                    size_t w_idx = sb * 8 + sub * 2 + w_in_sub;
                    if (w_idx < words_.size()) {
                        uint64_t w = words_[w_idx];
                        sub_counts[sub] += static_cast<uint32_t>(std::popcount(w));
                    }
                }
            }

            uint32_t delta1 = sub_counts[0];
            uint32_t delta2 = delta1 + sub_counts[1];
            uint32_t delta3 = delta2 + sub_counts[2];

            uint64_t packed_entry = detail::L1L2Entry::encode(l1, delta1, delta2, delta3).raw;
            l1l2_[sb] = packed_entry;

            for (size_t w_offset = 0; w_offset < 8; ++w_offset) {
                size_t w_idx = sb * 8 + w_offset;
                if (w_idx >= words_.size()) {
                    break;
                }
                uint64_t w = words_[w_idx];
                size_t bits_in_word = 64;
                if ((w_idx + 1) * 64 > num_bits_) {
                    bits_in_word = num_bits_ - (w_idx * 64);
                }

                uint32_t ones_in_w = static_cast<uint32_t>(std::popcount(w));
                uint32_t zeros_in_w = static_cast<uint32_t>(bits_in_word - ones_in_w);

                if (running_ones + ones_in_w >= next_select1_target) {
                    uint64_t tmp = w;
                    size_t bit_base = w_idx * 64;
                    while (tmp != 0) {
                        size_t bit = static_cast<size_t>(std::countr_zero(tmp));
                        if (bit >= bits_in_word) {
                            break;
                        }
                        size_t pos = bit_base + bit;
                        uint64_t one_rank =
                            running_ones + std::popcount(w & ((1ULL << bit) - 1ULL)) + 1;
                        if (one_rank >= next_select1_target) {
                            select1_samples_.push_back(static_cast<uint32_t>(pos));
                            next_select1_target += kSelectSampleRate;
                        }
                        tmp &= (tmp - 1ULL);
                    }
                }

                if (running_zeros + zeros_in_w >= next_select0_target) {
                    uint64_t tmp = (~w);
                    if (bits_in_word < 64) {
                        tmp &= (1ULL << bits_in_word) - 1ULL;
                    }
                    size_t bit_base = w_idx * 64;
                    while (tmp != 0) {
                        size_t bit = static_cast<size_t>(std::countr_zero(tmp));
                        if (bit >= bits_in_word) {
                            break;
                        }
                        size_t pos = bit_base + bit;
                        uint64_t zero_rank =
                            running_zeros + (bit - std::popcount(w & ((1ULL << bit) - 1ULL))) + 1;
                        if (zero_rank >= next_select0_target) {
                            select0_samples_.push_back(static_cast<uint32_t>(pos));
                            next_select0_target += kSelectSampleRate;
                        }
                        tmp &= (tmp - 1ULL);
                    }
                }

                running_ones += ones_in_w;
                running_zeros += zeros_in_w;
            }
        }

        total_ones_ = running_ones;
        select1_samples_.shrink_to_fit();
        select0_samples_.shrink_to_fit();
        l1l2_.shrink_to_fit();
        l0_.shrink_to_fit();
    }
};

} // namespace algoat::succinct

#pragma once

#include "algoat/succinct/rank_select.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define ALGOAT_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define ALGOAT_ALWAYS_INLINE __forceinline
#else
#define ALGOAT_ALWAYS_INLINE inline
#endif

namespace algoat::succinct {

namespace detail {

/// \brief Binary header for zero-copy serialization of Elias-Fano lists.
/// \details 64-byte aligned header storing parameters and word dimensions.
struct alignas(8) EliasFanoHeader {
    uint64_t magic{0x464554414F474C41ULL}; // 'ALGOATEF'
    uint32_t version{1};
    uint32_t lower_bits{0};
    uint64_t n{0};
    uint64_t universe{0};
    uint64_t lower_words_count{0};
    uint64_t upper_words_count{0};
    uint64_t upper_bits_count{0};
    uint64_t reserved{0}; // Pad to 64 bytes
};

static_assert(sizeof(EliasFanoHeader) == 64, "EliasFanoHeader must be exactly 64 bytes");

} // namespace detail

/// \brief Encodes a non-decreasing sequence into the Elias-Fano representation.
class EliasFanoEncoder {
public:
    /// \param n Number of integers to encode.
    /// \param universe Upper bound u (i.e., all values in [0, u)).
    EliasFanoEncoder(size_t n, uint64_t universe) : n_(n), universe_(universe) {
        if (universe_ == 0 && n_ > 0) {
            throw std::invalid_argument("Universe cannot be 0 for non-empty sequence");
        }

        if (n_ > 0 && universe_ > n_) {
            lower_bits_ = static_cast<size_t>(std::bit_width(universe_ / n_) - 1);
            if (lower_bits_ >= 64) {
                lower_bits_ = 63;
            }
        } else {
            lower_bits_ = 0;
        }

        if (n_ > 0 && lower_bits_ > 0) {
            size_t total_lower_bits = n_ * lower_bits_;
            size_t num_lower_words =
                (total_lower_bits + 63) / 64 + 4; // Padding for unaligned 64-bit loads
            lower_words_.resize(num_lower_words, 0ULL);
        }

        if (n_ > 0 && universe_ > 0) {
            uint64_t max_high = (universe_ - 1) >> lower_bits_;
            size_t estimated_upper_bits = n_ + max_high + 1;
            size_t num_upper_words = (estimated_upper_bits + 63) / 64 + 4;
            upper_words_.resize(num_upper_words, 0ULL);
        }
    }

    /// \brief Appends the next value (must be >= previous appended value).
    void push(uint64_t value) {
        assert(count_ < n_);
        assert(value < universe_);
        assert(count_ == 0 || value >= last_);

        // 1. Pack lower bits
        if (lower_bits_ > 0) {
            uint64_t low = value & ((1ULL << lower_bits_) - 1ULL);
            size_t bit_pos = count_ * lower_bits_;
            size_t w_idx = bit_pos >> 6;
            size_t b_off = bit_pos & 63;

            lower_words_[w_idx] |= (low << b_off);
            if (b_off + lower_bits_ > 64) {
                lower_words_[w_idx + 1] |= (low >> (64 - b_off));
            }
        }

        // 2. Set upper bits unary bit
        uint64_t high = value >> lower_bits_;
        size_t upper_pos = high + count_;
        size_t up_w_idx = upper_pos >> 6;
        size_t up_b_off = upper_pos & 63;

        if (up_w_idx >= upper_words_.size()) {
            upper_words_.resize(up_w_idx + 8, 0ULL);
        }
        upper_words_[up_w_idx] |= (1ULL << up_b_off);
        if (upper_pos + 1 > upper_size_) {
            upper_size_ = upper_pos + 1;
        }

        last_ = value;
        ++count_;
    }

    /// \brief Finalizes encoding. After this call, access via EliasFanoList.
    /// \returns Serialized byte buffer (owns memory).
    [[nodiscard]] std::vector<uint8_t> finalize() {
        assert(count_ == n_);

        size_t actual_lower_words = 0;
        if (n_ > 0 && lower_bits_ > 0) {
            size_t total_lower_bits = n_ * lower_bits_;
            actual_lower_words = (total_lower_bits + 63) / 64 + 4;
        }
        lower_words_.resize(actual_lower_words, 0ULL);

        size_t actual_upper_words = 0;
        if (n_ > 0 && upper_size_ > 0) {
            actual_upper_words = (upper_size_ + 63) / 64;
        }
        upper_words_.resize(actual_upper_words, 0ULL);

        detail::EliasFanoHeader header;
        header.lower_bits = static_cast<uint32_t>(lower_bits_);
        header.n = n_;
        header.universe = universe_;
        header.lower_words_count = actual_lower_words;
        header.upper_words_count = actual_upper_words;
        header.upper_bits_count = upper_size_;

        size_t total_bytes = sizeof(detail::EliasFanoHeader) +
                             (actual_lower_words * sizeof(uint64_t)) +
                             (actual_upper_words * sizeof(uint64_t));

        std::vector<uint8_t> buffer(total_bytes, 0);
        std::memcpy(buffer.data(), &header, sizeof(header));

        size_t offset = sizeof(header);
        if (actual_lower_words > 0) {
            size_t lower_bytes = actual_lower_words * sizeof(uint64_t);
            std::memcpy(buffer.data() + offset, lower_words_.data(), lower_bytes);
            offset += lower_bytes;
        }
        if (actual_upper_words > 0) {
            size_t upper_bytes = actual_upper_words * sizeof(uint64_t);
            std::memcpy(buffer.data() + offset, upper_words_.data(), upper_bytes);
        }

        return buffer;
    }

private:
    size_t n_;
    uint64_t universe_;
    size_t lower_bits_{0};
    size_t count_{0};
    uint64_t last_{0};

    // Dense lower-bits array packed at `lower_bits_` bits per element
    std::vector<uint64_t> lower_words_;

    // Upper unary sequence bitvector (bit set at position upper_val[i] + i)
    std::vector<uint64_t> upper_words_;
    size_t upper_size_{0};
};

/// \brief Immutable decoded view over an Elias-Fano encoded sequence.
class EliasFanoList {
public:
    class const_iterator;

    EliasFanoList() = default;

    /// \brief Constructs from a previously encoded buffer (borrows data when 8-byte aligned).
    explicit EliasFanoList(std::span<const uint8_t> data) {
        init_from_buffer(data, false);
    }

    /// \brief Constructs and owns data from a vector.
    explicit EliasFanoList(std::vector<uint8_t> data) {
        owned_data_ = std::move(data);
        init_from_buffer(std::span<const uint8_t>{owned_data_.data(), owned_data_.size()}, true);
    }

    EliasFanoList(const EliasFanoList& other)
        : n_(other.n_), universe_(other.universe_), lower_bits_(other.lower_bits_),
          lower_mask_(other.lower_mask_), lower_words_count_(other.lower_words_count_),
          upper_words_count_(other.upper_words_count_), upper_bits_(other.upper_bits_),
          owned_data_(other.owned_data_) {
        if (!owned_data_.empty()) {
            rebind_pointers();
        } else {
            lower_words_ = other.lower_words_;
            lower_bytes_ = other.lower_bytes_;
            upper_words_ = other.upper_words_;
        }
    }

    EliasFanoList(EliasFanoList&& other) noexcept
        : n_(other.n_), universe_(other.universe_), lower_bits_(other.lower_bits_),
          lower_mask_(other.lower_mask_), lower_words_count_(other.lower_words_count_),
          upper_words_count_(other.upper_words_count_), upper_bits_(std::move(other.upper_bits_)),
          owned_data_(std::move(other.owned_data_)) {
        if (!owned_data_.empty()) {
            rebind_pointers();
        } else {
            lower_words_ = other.lower_words_;
            lower_bytes_ = other.lower_bytes_;
            upper_words_ = other.upper_words_;
        }
        other.lower_words_ = nullptr;
        other.lower_bytes_ = nullptr;
        other.upper_words_ = nullptr;
        other.n_ = 0;
        other.universe_ = 0;
        other.lower_bits_ = 0;
        other.lower_mask_ = 0;
        other.lower_words_count_ = 0;
        other.upper_words_count_ = 0;
    }

    EliasFanoList& operator=(const EliasFanoList& other) {
        if (this != &other) {
            n_ = other.n_;
            universe_ = other.universe_;
            lower_bits_ = other.lower_bits_;
            lower_mask_ = other.lower_mask_;
            lower_words_count_ = other.lower_words_count_;
            upper_words_count_ = other.upper_words_count_;
            upper_bits_ = other.upper_bits_;
            owned_data_ = other.owned_data_;
            if (!owned_data_.empty()) {
                rebind_pointers();
            } else {
                lower_words_ = other.lower_words_;
                lower_bytes_ = other.lower_bytes_;
                upper_words_ = other.upper_words_;
            }
        }
        return *this;
    }

    EliasFanoList& operator=(EliasFanoList&& other) noexcept {
        if (this != &other) {
            n_ = other.n_;
            universe_ = other.universe_;
            lower_bits_ = other.lower_bits_;
            lower_mask_ = other.lower_mask_;
            lower_words_count_ = other.lower_words_count_;
            upper_words_count_ = other.upper_words_count_;
            upper_bits_ = std::move(other.upper_bits_);
            owned_data_ = std::move(other.owned_data_);
            if (!owned_data_.empty()) {
                rebind_pointers();
            } else {
                lower_words_ = other.lower_words_;
                lower_bytes_ = other.lower_bytes_;
                upper_words_ = other.upper_words_;
            }
            other.lower_words_ = nullptr;
            other.lower_bytes_ = nullptr;
            other.upper_words_ = nullptr;
            other.n_ = 0;
            other.universe_ = 0;
            other.lower_bits_ = 0;
            other.lower_mask_ = 0;
            other.lower_words_count_ = 0;
            other.upper_words_count_ = 0;
        }
        return *this;
    }

    ~EliasFanoList() = default;

    /// \brief Retrieves the i-th element (0-indexed) in O(1).
    [[nodiscard]] ALGOAT_ALWAYS_INLINE uint64_t at(size_t i) const noexcept {
        assert(i < n_);
        size_t pos = upper_bits_.select1(i + 1);
        uint64_t high = static_cast<uint64_t>(pos - i);
        if (lower_bits_ == 0) [[unlikely]] {
            return high;
        }
        return (high << lower_bits_) | extract_lower(i);
    }

    /// \brief Returns the position of the first value >= val.
    [[nodiscard]] size_t lower_bound_pos(uint64_t val) const noexcept {
        if (n_ == 0 || val >= universe_) {
            return n_;
        }
        if (val == 0) {
            return 0;
        }

        uint64_t high_quotient = val >> lower_bits_;
        uint64_t low_remainder = val & lower_mask_;

        uint64_t total_zeros = upper_bits_.bit_size() - upper_bits_.total_ones();
        if (high_quotient > total_zeros) {
            return n_;
        }

        size_t candidate_start = 0;
        size_t zero_pos = 0;
        if (high_quotient > 0) {
            zero_pos = upper_bits_.select0(high_quotient);
            candidate_start = zero_pos - high_quotient + 1;
        }

        if (candidate_start >= n_) {
            return n_;
        }

        // Fast path: inspect upper bit at (zero_pos + 1) directly
        if (high_quotient > 0) {
            size_t next_bit_pos = zero_pos + 1;
            if (next_bit_pos < upper_bits_.bit_size() && !upper_bits_.access(next_bit_pos)) {
                // Next bit is 0 -> bucket high_quotient has 0 elements!
                return candidate_start;
            }
        }

        // Check if first element in bucket satisfies lower bits
        if (extract_lower(candidate_start) >= low_remainder) {
            return candidate_start;
        }

        // Fast path for 1-element bucket: inspect bit (zero_pos + 2)
        if (high_quotient > 0) {
            size_t bit2_pos = zero_pos + 2;
            if (bit2_pos >= upper_bits_.bit_size() || !upper_bits_.access(bit2_pos)) {
                // Bucket high_quotient only had 1 element, and its value was < val -> next bucket
                // starts at candidate_start + 1
                return candidate_start + 1;
            }
        }

        size_t candidate_end = n_;
        if (high_quotient < total_zeros) {
            size_t p1 = upper_bits_.select0(high_quotient + 1);
            candidate_end = std::min(n_, static_cast<size_t>(p1 - high_quotient));
        }

        if (candidate_start == candidate_end) {
            return candidate_start;
        }

        size_t lo = candidate_start + 1;
        size_t hi = candidate_end;
        if (hi - lo <= 8) {
            for (size_t idx = lo; idx < hi; ++idx) {
                if (extract_lower(idx) >= low_remainder) {
                    return idx;
                }
            }
            return hi;
        } else {
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (extract_lower(mid) < low_remainder) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }
    }

    /// \brief Returns the smallest value >= val, or universe() if none exists.
    /// Used for inverted index intersection (WAND skipping).
    [[nodiscard]] ALGOAT_ALWAYS_INLINE uint64_t next_geq(uint64_t val) const noexcept {
        size_t pos = lower_bound_pos(val);
        return (pos < n_) ? at(pos) : universe_;
    }

    [[nodiscard]] ALGOAT_ALWAYS_INLINE size_t size() const noexcept {
        return n_;
    }

    [[nodiscard]] ALGOAT_ALWAYS_INLINE uint64_t universe() const noexcept {
        return universe_;
    }

    [[nodiscard]] ALGOAT_ALWAYS_INLINE size_t lower_bits() const noexcept {
        return lower_bits_;
    }

    [[nodiscard]] size_t memory_usage_bytes() const noexcept {
        size_t lower_bytes = (lower_words_count_ * sizeof(uint64_t));
        size_t upper_raw_bytes = (upper_bits_.bit_size() + 7) / 8;
        return lower_bytes + upper_raw_bytes + upper_bits_.total_index_bytes() +
               sizeof(detail::EliasFanoHeader);
    }

    /// \brief High-performance streaming iterator over the Elias-Fano sequence.
    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = uint64_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const uint64_t*;
        using reference = uint64_t;

        const_iterator() noexcept = default;

        [[nodiscard]] ALGOAT_ALWAYS_INLINE reference operator*() const noexcept {
            return curr_val_;
        }

        [[nodiscard]] ALGOAT_ALWAYS_INLINE pointer operator->() const noexcept {
            return &curr_val_;
        }

        ALGOAT_ALWAYS_INLINE const_iterator& operator++() noexcept {
            assert(i_ < n_);
            ++i_;
            if (i_ < n_) [[likely]] {
                advance_one();
            }
            return *this;
        }

        ALGOAT_ALWAYS_INLINE const_iterator operator++(int) noexcept {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        [[nodiscard]] ALGOAT_ALWAYS_INLINE bool
        operator==(const const_iterator& other) const noexcept {
            return i_ == other.i_;
        }

        [[nodiscard]] ALGOAT_ALWAYS_INLINE bool
        operator!=(const const_iterator& other) const noexcept {
            return i_ != other.i_;
        }

    private:
        friend class EliasFanoList;

        const_iterator(const EliasFanoList* list, size_t index) : i_(index) {
            if (list) {
                n_ = list->n_;
                lower_bits_ = list->lower_bits_;
                lower_mask_ = list->lower_mask_;
                upper_ptr_ = list->upper_words_;
                upper_words_count_ = list->upper_words_count_;
                lower_bytes_ptr_ = list->lower_bytes_;
                if (i_ < n_) {
                    seek(i_, list);
                }
            }
        }

        void seek(size_t index, const EliasFanoList* list) noexcept {
            i_ = index;
            if (i_ >= n_) {
                return;
            }

            size_t target_1_bit = list->upper_bits_.select1(index + 1);
            up_w_idx_ = target_1_bit >> 6;
            uint32_t pos = static_cast<uint32_t>(target_1_bit & 63);

            curr_high_ = static_cast<uint64_t>(target_1_bit - index);
            last_pos_ = static_cast<int32_t>(pos);
            uint64_t w = upper_ptr_[up_w_idx_];
            curr_up_word_ = (pos == 63) ? 0ULL : (w & ~((2ULL << pos) - 1ULL));

            bit_pos_ = index * lower_bits_;
            if (lower_bits_ > 0) [[likely]] {
                uint64_t raw_w;
                std::memcpy(&raw_w, lower_bytes_ptr_ + (bit_pos_ >> 3), sizeof(raw_w));
                uint64_t low = (raw_w >> (bit_pos_ & 7)) & lower_mask_;
                bit_pos_ += lower_bits_;
                curr_val_ = (curr_high_ << lower_bits_) | low;
            } else {
                curr_val_ = curr_high_;
            }
        }

        ALGOAT_ALWAYS_INLINE void advance_one() noexcept {
            while (curr_up_word_ == 0ULL) {
                curr_high_ += static_cast<uint64_t>(63 - last_pos_);
                ++up_w_idx_;
                if (up_w_idx_ >= upper_words_count_) [[unlikely]] {
                    return;
                }
                curr_up_word_ = upper_ptr_[up_w_idx_];
                last_pos_ = -1;
            }

            uint32_t pos = static_cast<uint32_t>(std::countr_zero(curr_up_word_));
            curr_high_ += static_cast<uint64_t>(pos - last_pos_ - 1);
            last_pos_ = static_cast<int32_t>(pos);
            curr_up_word_ &= (curr_up_word_ - 1ULL);

            if (lower_bits_ > 0) [[likely]] {
                uint64_t raw_w;
                std::memcpy(&raw_w, lower_bytes_ptr_ + (bit_pos_ >> 3), sizeof(raw_w));
                uint64_t low = (raw_w >> (bit_pos_ & 7)) & lower_mask_;
                bit_pos_ += lower_bits_;
                curr_val_ = (curr_high_ << lower_bits_) | low;
            } else {
                curr_val_ = curr_high_;
            }
        }

        size_t i_{0};
        size_t n_{0};
        size_t lower_bits_{0};
        uint64_t lower_mask_{0};
        uint64_t curr_val_{0};

        const uint64_t* upper_ptr_{nullptr};
        const uint8_t* lower_bytes_ptr_{nullptr};
        size_t up_w_idx_{0};
        size_t upper_words_count_{0};
        uint64_t curr_up_word_{0};
        uint64_t curr_high_{0};
        int32_t last_pos_{-1};
        size_t bit_pos_{0};
    };

    [[nodiscard]] ALGOAT_ALWAYS_INLINE const_iterator begin() const noexcept {
        return const_iterator(this, 0);
    }

    [[nodiscard]] ALGOAT_ALWAYS_INLINE const_iterator end() const noexcept {
        return const_iterator(this, n_);
    }

private:
    size_t n_{0};
    uint64_t universe_{0};
    size_t lower_bits_{0};
    uint64_t lower_mask_{0};
    size_t lower_words_count_{0};
    size_t upper_words_count_{0};

    const uint64_t* lower_words_{nullptr};
    const uint8_t* lower_bytes_{nullptr};
    const uint64_t* upper_words_{nullptr};
    RankSelectBitVector upper_bits_; // Backed by Issue #20 implementation

    std::vector<uint8_t> owned_data_;

    void rebind_pointers() noexcept {
        if (!owned_data_.empty()) {
            size_t offset = sizeof(detail::EliasFanoHeader);
            if (lower_words_count_ > 0) {
                lower_words_ = reinterpret_cast<const uint64_t*>(owned_data_.data() + offset);
                lower_bytes_ = owned_data_.data() + offset;
            }
            offset += lower_words_count_ * sizeof(uint64_t);
            if (upper_words_count_ > 0) {
                upper_words_ = reinterpret_cast<const uint64_t*>(owned_data_.data() + offset);
            }
        }
    }

    [[nodiscard]] ALGOAT_ALWAYS_INLINE uint64_t extract_lower(size_t i) const noexcept {
        if (lower_bits_ == 0 || !lower_bytes_) [[unlikely]] {
            return 0;
        }
        size_t bit_pos = i * lower_bits_;
        uint64_t raw_w = 0;
        std::memcpy(&raw_w, lower_bytes_ + (bit_pos >> 3), sizeof(raw_w));
        return (raw_w >> (bit_pos & 7)) & lower_mask_;
    }

    void init_from_buffer(std::span<const uint8_t> buffer, bool owns) {
        if (buffer.size() < sizeof(detail::EliasFanoHeader)) {
            throw std::invalid_argument("Serialized Elias-Fano buffer smaller than header size");
        }

        detail::EliasFanoHeader header;
        std::memcpy(&header, buffer.data(), sizeof(header));
        if (header.magic != 0x464554414F474C41ULL || header.version != 1) {
            throw std::invalid_argument("Invalid Elias-Fano serialized buffer header");
        }

        n_ = header.n;
        universe_ = header.universe;
        lower_bits_ = header.lower_bits;
        lower_mask_ = (lower_bits_ > 0) ? ((1ULL << lower_bits_) - 1ULL) : 0ULL;
        lower_words_count_ = header.lower_words_count;
        upper_words_count_ = header.upper_words_count;

        if (n_ == 0) {
            return;
        }

        size_t offset = sizeof(detail::EliasFanoHeader);
        size_t lower_bytes = header.lower_words_count * sizeof(uint64_t);
        size_t upper_bytes = header.upper_words_count * sizeof(uint64_t);

        if (buffer.size() < offset + lower_bytes + upper_bytes) {
            throw std::invalid_argument("Buffer underflow in Elias-Fano deserialization");
        }

        bool is_aligned = (reinterpret_cast<uintptr_t>(buffer.data()) % alignof(uint64_t) == 0);
        if (!owns && !is_aligned) {
            owned_data_.assign(buffer.begin(), buffer.end());
            owns = true;
            buffer = std::span<const uint8_t>{owned_data_.data(), owned_data_.size()};
        }

        if (header.lower_words_count > 0) {
            lower_words_ = reinterpret_cast<const uint64_t*>(buffer.data() + offset);
            lower_bytes_ = buffer.data() + offset;
        }
        offset += lower_bytes;

        if (header.upper_words_count > 0 && header.upper_bits_count > 0) {
            upper_words_ = reinterpret_cast<const uint64_t*>(buffer.data() + offset);
            upper_bits_ = RankSelectBitVector(
                std::span<const uint64_t>{upper_words_, header.upper_words_count},
                header.upper_bits_count);
        }
    }
};

} // namespace algoat::succinct
